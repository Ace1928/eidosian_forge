from __future__ import absolute_import
from _ruamel_yaml import CParser, CEmitter  # type: ignore
from ruamel.yaml.constructor import Constructor, BaseConstructor, SafeConstructor
from ruamel.yaml.representer import Representer, SafeRepresenter, BaseRepresenter
from ruamel.yaml.resolver import Resolver, BaseResolver
class CBaseLoader(CParser, BaseConstructor, BaseResolver):

    def __init__(self, stream, version=None, preserve_quotes=None):
        CParser.__init__(self, stream)
        self._parser = self._composer = self
        BaseConstructor.__init__(self, loader=self)
        BaseResolver.__init__(self, loadumper=self)