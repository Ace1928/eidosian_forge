from __future__ import annotations
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import nprint, dbg, DBG_EVENT, \
class RoundTripEmitter(Emitter):

    def prepare_tag(self, ctag: Any) -> Any:
        if not ctag:
            raise EmitterError('tag must not be empty')
        tag = str(ctag)
        if tag == '!' or tag == '!!':
            return tag
        handle = ctag.handle
        suffix = ctag.suffix
        prefixes = sorted(self.tag_prefixes.keys())
        if handle is None:
            for prefix in prefixes:
                if tag.startswith(prefix) and (prefix == '!' or len(prefix) < len(tag)):
                    handle = self.tag_prefixes[prefix]
                    suffix = suffix[len(prefix):]
        if handle:
            return f'{handle!s}{suffix!s}'
        else:
            return f'!<{suffix!s}>'