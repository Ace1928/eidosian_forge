from __future__ import absolute_import
from _ruamel_yaml import CParser, CEmitter  # type: ignore
from ruamel.yaml.constructor import Constructor, BaseConstructor, SafeConstructor
from ruamel.yaml.representer import Representer, SafeRepresenter, BaseRepresenter
from ruamel.yaml.resolver import Resolver, BaseResolver
class CDumper(CEmitter, Representer, Resolver):

    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None, allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None, version=None, tags=None, block_seq_indent=None, top_level_colon_align=None, prefix_colon=None):
        CEmitter.__init__(self, stream, canonical=canonical, indent=indent, width=width, encoding=encoding, allow_unicode=allow_unicode, line_break=line_break, explicit_start=explicit_start, explicit_end=explicit_end, version=version, tags=tags)
        self._emitter = self._serializer = self._representer = self
        Representer.__init__(self, default_style=default_style, default_flow_style=default_flow_style)
        Resolver.__init__(self)