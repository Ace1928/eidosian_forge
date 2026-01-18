from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _context_from_explicit(self, object_type, local_conf, global_conf, global_addition, section):
    possible = []
    for protocol_options in object_type.egg_protocols:
        for protocol in protocol_options:
            if protocol in local_conf:
                possible.append((protocol, local_conf[protocol]))
                break
    if len(possible) > 1:
        raise LookupError(f'Multiple protocols given in section {section!r}: {possible}')
    if not possible:
        raise LookupError('No loader given in section %r' % section)
    found_protocol, found_expr = possible[0]
    del local_conf[found_protocol]
    value = importlib_metadata.EntryPoint(name=None, group=None, value=found_expr).load()
    context = LoaderContext(value, object_type, found_protocol, global_conf, local_conf, self)
    return context