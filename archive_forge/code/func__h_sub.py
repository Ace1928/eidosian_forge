import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _h_sub(self, match):
    """Handles processing markdown headers"""
    if match.group(1) is not None and match.group(3) == '-':
        return match.group(1)
    elif match.group(1) is not None:
        n = {'=': 1, '-': 2}[match.group(3)[0]]
        header_group = match.group(2)
    else:
        n = len(match.group(5))
        header_group = match.group(6)
    demote_headers = self.extras.get('demote-headers')
    if demote_headers:
        n = min(n + demote_headers, 6)
    header_id_attr = ''
    if 'header-ids' in self.extras:
        header_id = self.header_id_from_text(header_group, self.extras['header-ids'].get('prefix'), n)
        if header_id:
            header_id_attr = ' id="%s"' % header_id
    html = self._run_span_gamut(header_group)
    if 'toc' in self.extras and header_id:
        self._toc_add_entry(n, header_id, html)
    return '<h%d%s>%s</h%d>\n\n' % (n, header_id_attr, html, n)