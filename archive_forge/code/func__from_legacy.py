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
def _from_legacy(self):
    assert self._legacy and (not self._data)
    result = {'metadata_version': self.METADATA_VERSION, 'generator': self.GENERATOR}
    lmd = self._legacy.todict(True)
    for k in ('name', 'version', 'license', 'summary', 'description', 'classifier'):
        if k in lmd:
            if k == 'classifier':
                nk = 'classifiers'
            else:
                nk = k
            result[nk] = lmd[k]
    kw = lmd.get('Keywords', [])
    if kw == ['']:
        kw = []
    result['keywords'] = kw
    keys = (('requires_dist', 'run_requires'), ('setup_requires_dist', 'build_requires'))
    for ok, nk in keys:
        if ok in lmd and lmd[ok]:
            result[nk] = [{'requires': lmd[ok]}]
    result['provides'] = self.provides
    author = {}
    maintainer = {}
    return result