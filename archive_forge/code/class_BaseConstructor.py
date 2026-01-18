from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
class BaseConstructor(object):
    yaml_constructors = {}
    yaml_multi_constructors = {}

    def __init__(self, preserve_quotes=None, loader=None):
        self.loader = loader
        if self.loader is not None and getattr(self.loader, '_constructor', None) is None:
            self.loader._constructor = self
        self.loader = loader
        self.yaml_base_dict_type = dict
        self.yaml_base_list_type = list
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.state_generators = []
        self.deep_construct = False
        self._preserve_quotes = preserve_quotes
        self.allow_duplicate_keys = version_tnf((0, 15, 1), (0, 16))

    @property
    def composer(self):
        if hasattr(self.loader, 'typ'):
            return self.loader.composer
        try:
            return self.loader._composer
        except AttributeError:
            sys.stdout.write('slt {}\n'.format(type(self)))
            sys.stdout.write('slc {}\n'.format(self.loader._composer))
            sys.stdout.write('{}\n'.format(dir(self)))
            raise

    @property
    def resolver(self):
        if hasattr(self.loader, 'typ'):
            return self.loader.resolver
        return self.loader._resolver

    def check_data(self):
        return self.composer.check_node()

    def get_data(self):
        if self.composer.check_node():
            return self.construct_document(self.composer.get_node())

    def get_single_data(self):
        node = self.composer.get_single_node()
        if node is not None:
            return self.construct_document(node)
        return None

    def construct_document(self, node):
        data = self.construct_object(node)
        while bool(self.state_generators):
            state_generators = self.state_generators
            self.state_generators = []
            for generator in state_generators:
                for _dummy in generator:
                    pass
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.deep_construct = False
        return data

    def construct_object(self, node, deep=False):
        """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
        if node in self.constructed_objects:
            return self.constructed_objects[node]
        if deep:
            old_deep = self.deep_construct
            self.deep_construct = True
        if node in self.recursive_objects:
            return self.recursive_objects[node]
        self.recursive_objects[node] = None
        constructor = None
        tag_suffix = None
        if node.tag in self.yaml_constructors:
            constructor = self.yaml_constructors[node.tag]
        else:
            for tag_prefix in self.yaml_multi_constructors:
                if node.tag.startswith(tag_prefix):
                    tag_suffix = node.tag[len(tag_prefix):]
                    constructor = self.yaml_multi_constructors[tag_prefix]
                    break
            else:
                if None in self.yaml_multi_constructors:
                    tag_suffix = node.tag
                    constructor = self.yaml_multi_constructors[None]
                elif None in self.yaml_constructors:
                    constructor = self.yaml_constructors[None]
                elif isinstance(node, ScalarNode):
                    constructor = self.__class__.construct_scalar
                elif isinstance(node, SequenceNode):
                    constructor = self.__class__.construct_sequence
                elif isinstance(node, MappingNode):
                    constructor = self.__class__.construct_mapping
        if tag_suffix is None:
            data = constructor(self, node)
        else:
            data = constructor(self, tag_suffix, node)
        if isinstance(data, types.GeneratorType):
            generator = data
            data = next(generator)
            if self.deep_construct:
                for _dummy in generator:
                    pass
            else:
                self.state_generators.append(generator)
        self.constructed_objects[node] = data
        del self.recursive_objects[node]
        if deep:
            self.deep_construct = old_deep
        return data

    def construct_scalar(self, node):
        if not isinstance(node, ScalarNode):
            raise ConstructorError(None, None, 'expected a scalar node, but found %s' % node.id, node.start_mark)
        return node.value

    def construct_sequence(self, node, deep=False):
        """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
        if not isinstance(node, SequenceNode):
            raise ConstructorError(None, None, 'expected a sequence node, but found %s' % node.id, node.start_mark)
        return [self.construct_object(child, deep=deep) for child in node.value]

    def construct_mapping(self, node, deep=False):
        """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        total_mapping = self.yaml_base_dict_type()
        if getattr(node, 'merge', None) is not None:
            todo = [(node.merge, False), (node.value, False)]
        else:
            todo = [(node.value, True)]
        for values, check in todo:
            mapping = self.yaml_base_dict_type()
            for key_node, value_node in values:
                key = self.construct_object(key_node, deep=True)
                if not isinstance(key, Hashable):
                    if isinstance(key, list):
                        key = tuple(key)
                if PY2:
                    try:
                        hash(key)
                    except TypeError as exc:
                        raise ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
                elif not isinstance(key, Hashable):
                    raise ConstructorError('while constructing a mapping', node.start_mark, 'found unhashable key', key_node.start_mark)
                value = self.construct_object(value_node, deep=deep)
                if check:
                    if self.check_mapping_key(node, key_node, mapping, key, value):
                        mapping[key] = value
                else:
                    mapping[key] = value
            total_mapping.update(mapping)
        return total_mapping

    def check_mapping_key(self, node, key_node, mapping, key, value):
        """return True if key is unique"""
        if key in mapping:
            if not self.allow_duplicate_keys:
                mk = mapping.get(key)
                if PY2:
                    if isinstance(key, unicode):
                        key = key.encode('utf-8')
                    if isinstance(value, unicode):
                        value = value.encode('utf-8')
                    if isinstance(mk, unicode):
                        mk = mk.encode('utf-8')
                args = ['while constructing a mapping', node.start_mark, 'found duplicate key "{}" with value "{}" (original value: "{}")'.format(key, value, mk), key_node.start_mark, '\n                    To suppress this check see:\n                        http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                    ', '                    Duplicate keys will become an error in future releases, and are errors\n                    by default when using the new API.\n                    ']
                if self.allow_duplicate_keys is None:
                    warnings.warn(DuplicateKeyFutureWarning(*args))
                else:
                    raise DuplicateKeyError(*args)
            return False
        return True

    def check_set_key(self, node, key_node, setting, key):
        if key in setting:
            if not self.allow_duplicate_keys:
                if PY2:
                    if isinstance(key, unicode):
                        key = key.encode('utf-8')
                args = ['while constructing a set', node.start_mark, 'found duplicate key "{}"'.format(key), key_node.start_mark, '\n                    To suppress this check see:\n                        http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                    ', '                    Duplicate keys will become an error in future releases, and are errors\n                    by default when using the new API.\n                    ']
                if self.allow_duplicate_keys is None:
                    warnings.warn(DuplicateKeyFutureWarning(*args))
                else:
                    raise DuplicateKeyError(*args)

    def construct_pairs(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        pairs = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            pairs.append((key, value))
        return pairs

    @classmethod
    def add_constructor(cls, tag, constructor):
        if 'yaml_constructors' not in cls.__dict__:
            cls.yaml_constructors = cls.yaml_constructors.copy()
        cls.yaml_constructors[tag] = constructor

    @classmethod
    def add_multi_constructor(cls, tag_prefix, multi_constructor):
        if 'yaml_multi_constructors' not in cls.__dict__:
            cls.yaml_multi_constructors = cls.yaml_multi_constructors.copy()
        cls.yaml_multi_constructors[tag_prefix] = multi_constructor