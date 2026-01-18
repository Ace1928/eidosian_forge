import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _h_tag_sub(self, match):
    """Different to `_h_sub` in that this function handles existing HTML headers"""
    text = match.string[match.start():match.end()]
    h_level = int(match.group(1))
    id_attr = re.match('.*?id=(\\S+)?.*', match.group(2) or '') or ''
    if id_attr:
        id_attr = id_attr.group(1) or ''
    id_attr = id_attr.strip('\'" ')
    h_text = match.group(3)
    if id_attr and self._header_id_exists(id_attr):
        return text
    header_id = id_attr or self.header_id_from_text(h_text, self.extras['header-ids'].get('prefix'), h_level)
    if 'toc' in self.extras:
        self._toc_add_entry(h_level, header_id, h_text)
    if header_id and (not id_attr):
        return text[:3] + ' id="%s"' % header_id + text[3:]
    return text