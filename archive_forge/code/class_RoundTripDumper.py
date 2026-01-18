from __future__ import absolute_import
from ruamel.yaml.emitter import Emitter
from ruamel.yaml.serializer import Serializer
from ruamel.yaml.representer import (
from ruamel.yaml.resolver import Resolver, BaseResolver, VersionedResolver
class RoundTripDumper(Emitter, Serializer, RoundTripRepresenter, VersionedResolver):

    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None, allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None, version=None, tags=None, block_seq_indent=None, top_level_colon_align=None, prefix_colon=None):
        Emitter.__init__(self, stream, canonical=canonical, indent=indent, width=width, allow_unicode=allow_unicode, line_break=line_break, block_seq_indent=block_seq_indent, top_level_colon_align=top_level_colon_align, prefix_colon=prefix_colon, dumper=self)
        Serializer.__init__(self, encoding=encoding, explicit_start=explicit_start, explicit_end=explicit_end, version=version, tags=tags, dumper=self)
        RoundTripRepresenter.__init__(self, default_style=default_style, default_flow_style=default_flow_style, dumper=self)
        VersionedResolver.__init__(self, loader=self)