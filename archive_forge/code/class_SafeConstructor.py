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
class SafeConstructor(BaseConstructor):

    def construct_scalar(self, node):
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if key_node.tag == u'tag:yaml.org,2002:value':
                    return self.construct_scalar(value_node)
        return BaseConstructor.construct_scalar(self, node)

    def flatten_mapping(self, node):
        """
        This implements the merge key feature http://yaml.org/type/merge.html
        by inserting keys from the merge dict/list of dicts if not yet
        available in this node
        """
        merge = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == u'tag:yaml.org,2002:merge':
                if merge:
                    if self.allow_duplicate_keys:
                        del node.value[index]
                        index += 1
                        continue
                    args = ['while constructing a mapping', node.start_mark, 'found duplicate key "{}"'.format(key_node.value), key_node.start_mark, '\n                        To suppress this check see:\n                           http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                        ', '                        Duplicate keys will become an error in future releases, and are errors\n                        by default when using the new API.\n                        ']
                    if self.allow_duplicate_keys is None:
                        warnings.warn(DuplicateKeyFutureWarning(*args))
                    else:
                        raise DuplicateKeyError(*args)
                del node.value[index]
                if isinstance(value_node, MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, SequenceNode):
                    submerge = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, MappingNode):
                            raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping for merging, but found %s' % subnode.id, subnode.start_mark)
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    submerge.reverse()
                    for value in submerge:
                        merge.extend(value)
                else:
                    raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping or list of mappings for merging, but found %s' % value_node.id, value_node.start_mark)
            elif key_node.tag == u'tag:yaml.org,2002:value':
                key_node.tag = u'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1
        if bool(merge):
            node.merge = merge
            node.value = merge + node.value

    def construct_mapping(self, node, deep=False):
        """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        return BaseConstructor.construct_mapping(self, node, deep=deep)

    def construct_yaml_null(self, node):
        self.construct_scalar(node)
        return None
    bool_values = {u'yes': True, u'no': False, u'y': True, u'n': False, u'true': True, u'false': False, u'on': True, u'off': False}

    def construct_yaml_bool(self, node):
        value = self.construct_scalar(node)
        return self.bool_values[value.lower()]

    def construct_yaml_int(self, node):
        value_s = to_str(self.construct_scalar(node))
        value_s = value_s.replace('_', '')
        sign = +1
        if value_s[0] == '-':
            sign = -1
        if value_s[0] in '+-':
            value_s = value_s[1:]
        if value_s == '0':
            return 0
        elif value_s.startswith('0b'):
            return sign * int(value_s[2:], 2)
        elif value_s.startswith('0x'):
            return sign * int(value_s[2:], 16)
        elif value_s.startswith('0o'):
            return sign * int(value_s[2:], 8)
        elif self.resolver.processing_version == (1, 1) and value_s[0] == '0':
            return sign * int(value_s, 8)
        elif self.resolver.processing_version == (1, 1) and ':' in value_s:
            digits = [int(part) for part in value_s.split(':')]
            digits.reverse()
            base = 1
            value = 0
            for digit in digits:
                value += digit * base
                base *= 60
            return sign * value
        else:
            return sign * int(value_s)
    inf_value = 1e+300
    while inf_value != inf_value * inf_value:
        inf_value *= inf_value
    nan_value = -inf_value / inf_value

    def construct_yaml_float(self, node):
        value_so = to_str(self.construct_scalar(node))
        value_s = value_so.replace('_', '').lower()
        sign = +1
        if value_s[0] == '-':
            sign = -1
        if value_s[0] in '+-':
            value_s = value_s[1:]
        if value_s == '.inf':
            return sign * self.inf_value
        elif value_s == '.nan':
            return self.nan_value
        elif self.resolver.processing_version != (1, 2) and ':' in value_s:
            digits = [float(part) for part in value_s.split(':')]
            digits.reverse()
            base = 1
            value = 0.0
            for digit in digits:
                value += digit * base
                base *= 60
            return sign * value
        else:
            if self.resolver.processing_version != (1, 2) and 'e' in value_s:
                mantissa, exponent = value_s.split('e')
                if '.' not in mantissa:
                    warnings.warn(MantissaNoDotYAML1_1Warning(node, value_so))
            return sign * float(value_s)
    if PY3:

        def construct_yaml_binary(self, node):
            try:
                value = self.construct_scalar(node).encode('ascii')
            except UnicodeEncodeError as exc:
                raise ConstructorError(None, None, 'failed to convert base64 data into ascii: %s' % exc, node.start_mark)
            try:
                if hasattr(base64, 'decodebytes'):
                    return base64.decodebytes(value)
                else:
                    return base64.decodestring(value)
            except binascii.Error as exc:
                raise ConstructorError(None, None, 'failed to decode base64 data: %s' % exc, node.start_mark)
    else:

        def construct_yaml_binary(self, node):
            value = self.construct_scalar(node)
            try:
                return to_str(value).decode('base64')
            except (binascii.Error, UnicodeEncodeError) as exc:
                raise ConstructorError(None, None, 'failed to decode base64 data: %s' % exc, node.start_mark)
    timestamp_regexp = RegExp(u'^(?P<year>[0-9][0-9][0-9][0-9])\n          -(?P<month>[0-9][0-9]?)\n          -(?P<day>[0-9][0-9]?)\n          (?:((?P<t>[Tt])|[ \\t]+)   # explictly not retaining extra spaces\n          (?P<hour>[0-9][0-9]?)\n          :(?P<minute>[0-9][0-9])\n          :(?P<second>[0-9][0-9])\n          (?:\\.(?P<fraction>[0-9]*))?\n          (?:[ \\t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)\n          (?::(?P<tz_minute>[0-9][0-9]))?))?)?$', re.X)

    def construct_yaml_timestamp(self, node, values=None):
        if values is None:
            try:
                match = self.timestamp_regexp.match(node.value)
            except TypeError:
                match = None
            if match is None:
                raise ConstructorError(None, None, 'failed to construct timestamp from "{}"'.format(node.value), node.start_mark)
            values = match.groupdict()
        year = int(values['year'])
        month = int(values['month'])
        day = int(values['day'])
        if not values['hour']:
            return datetime.date(year, month, day)
        hour = int(values['hour'])
        minute = int(values['minute'])
        second = int(values['second'])
        fraction = 0
        if values['fraction']:
            fraction_s = values['fraction'][:6]
            while len(fraction_s) < 6:
                fraction_s += '0'
            fraction = int(fraction_s)
            if len(values['fraction']) > 6 and int(values['fraction'][6]) > 4:
                fraction += 1
        delta = None
        if values['tz_sign']:
            tz_hour = int(values['tz_hour'])
            minutes = values['tz_minute']
            tz_minute = int(minutes) if minutes else 0
            delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
            if values['tz_sign'] == '-':
                delta = -delta
        data = datetime.datetime(year, month, day, hour, minute, second, fraction)
        if delta:
            data -= delta
        return data

    def construct_yaml_omap(self, node):
        omap = ordereddict()
        yield omap
        if not isinstance(node, SequenceNode):
            raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a sequence, but found %s' % node.id, node.start_mark)
        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a mapping of length 1, but found %s' % subnode.id, subnode.start_mark)
            if len(subnode.value) != 1:
                raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a single mapping item, but found %d items' % len(subnode.value), subnode.start_mark)
            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            assert key not in omap
            value = self.construct_object(value_node)
            omap[key] = value

    def construct_yaml_pairs(self, node):
        pairs = []
        yield pairs
        if not isinstance(node, SequenceNode):
            raise ConstructorError('while constructing pairs', node.start_mark, 'expected a sequence, but found %s' % node.id, node.start_mark)
        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError('while constructing pairs', node.start_mark, 'expected a mapping of length 1, but found %s' % subnode.id, subnode.start_mark)
            if len(subnode.value) != 1:
                raise ConstructorError('while constructing pairs', node.start_mark, 'expected a single mapping item, but found %d items' % len(subnode.value), subnode.start_mark)
            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            value = self.construct_object(value_node)
            pairs.append((key, value))

    def construct_yaml_set(self, node):
        data = set()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_yaml_str(self, node):
        value = self.construct_scalar(node)
        if PY3:
            return value
        try:
            return value.encode('ascii')
        except UnicodeEncodeError:
            return value

    def construct_yaml_seq(self, node):
        data = self.yaml_base_list_type()
        yield data
        data.extend(self.construct_sequence(node))

    def construct_yaml_map(self, node):
        data = self.yaml_base_dict_type()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_yaml_object(self, node, cls):
        data = cls.__new__(cls)
        yield data
        if hasattr(data, '__setstate__'):
            state = self.construct_mapping(node, deep=True)
            data.__setstate__(state)
        else:
            state = self.construct_mapping(node)
            data.__dict__.update(state)

    def construct_undefined(self, node):
        raise ConstructorError(None, None, 'could not determine a constructor for the tag %r' % utf8(node.tag), node.start_mark)