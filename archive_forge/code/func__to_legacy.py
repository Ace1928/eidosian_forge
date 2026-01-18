from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def _to_legacy(self):

    def process_entries(entries):
        reqts = set()
        for e in entries:
            extra = e.get('extra')
            env = e.get('environment')
            rlist = e['requires']
            for r in rlist:
                if not env and (not extra):
                    reqts.add(r)
                else:
                    marker = ''
                    if extra:
                        marker = 'extra == "%s"' % extra
                    if env:
                        if marker:
                            marker = '(%s) and %s' % (env, marker)
                        else:
                            marker = env
                    reqts.add(';'.join((r, marker)))
        return reqts
    assert self._data and (not self._legacy)
    result = LegacyMetadata()
    nmd = self._data
    for nk, ok in self.LEGACY_MAPPING.items():
        if not isinstance(nk, tuple):
            if nk in nmd:
                result[ok] = nmd[nk]
        else:
            d = nmd
            found = True
            for k in nk:
                try:
                    d = d[k]
                except (KeyError, IndexError):
                    found = False
                    break
            if found:
                result[ok] = d
    r1 = process_entries(self.run_requires + self.meta_requires)
    r2 = process_entries(self.build_requires + self.dev_requires)
    if self.extras:
        result['Provides-Extra'] = sorted(self.extras)
    result['Requires-Dist'] = sorted(r1)
    result['Setup-Requires-Dist'] = sorted(r2)
    return result