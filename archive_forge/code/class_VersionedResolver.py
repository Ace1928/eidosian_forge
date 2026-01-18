from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
class VersionedResolver(BaseResolver):
    """
    contrary to the "normal" resolver, the smart resolver delays loading
    the pattern matching rules. That way it can decide to load 1.1 rules
    or the (default) 1.2 rules, that no longer support octal without 0o, sexagesimals
    and Yes/No/On/Off booleans.
    """

    def __init__(self, version=None, loader=None, loadumper=None):
        if loader is None and loadumper is not None:
            loader = loadumper
        BaseResolver.__init__(self, loader)
        self._loader_version = self.get_loader_version(version)
        self._version_implicit_resolver = {}

    def add_version_implicit_resolver(self, version, tag, regexp, first):
        if first is None:
            first = [None]
        impl_resolver = self._version_implicit_resolver.setdefault(version, {})
        for ch in first:
            impl_resolver.setdefault(ch, []).append((tag, regexp))

    def get_loader_version(self, version):
        if version is None or isinstance(version, tuple):
            return version
        if isinstance(version, list):
            return tuple(version)
        return tuple(map(int, version.split(u'.')))

    @property
    def versioned_resolver(self):
        """
        select the resolver based on the version we are parsing
        """
        version = self.processing_version
        if version not in self._version_implicit_resolver:
            for x in implicit_resolvers:
                if version in x[0]:
                    self.add_version_implicit_resolver(version, x[1], x[2], x[3])
        return self._version_implicit_resolver[version]

    def resolve(self, kind, value, implicit):
        if kind is ScalarNode and implicit[0]:
            if value == '':
                resolvers = self.versioned_resolver.get('', [])
            else:
                resolvers = self.versioned_resolver.get(value[0], [])
            resolvers += self.versioned_resolver.get(None, [])
            for tag, regexp in resolvers:
                if regexp.match(value):
                    return tag
            implicit = implicit[1]
        if bool(self.yaml_path_resolvers):
            exact_paths = self.resolver_exact_paths[-1]
            if kind in exact_paths:
                return exact_paths[kind]
            if None in exact_paths:
                return exact_paths[None]
        if kind is ScalarNode:
            return self.DEFAULT_SCALAR_TAG
        elif kind is SequenceNode:
            return self.DEFAULT_SEQUENCE_TAG
        elif kind is MappingNode:
            return self.DEFAULT_MAPPING_TAG

    @property
    def processing_version(self):
        try:
            version = self.parser.yaml_version
        except AttributeError:
            try:
                if hasattr(self.loadumper, 'typ'):
                    version = self.loadumper.version
                else:
                    version = self.loadumper._serializer.use_version
            except AttributeError:
                version = None
        if version is None:
            version = self._loader_version
            if version is None:
                version = _DEFAULT_YAML_VERSION
        return version