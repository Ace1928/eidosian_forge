import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _hash_html_block_sub(self, match, raw=False):
    if isinstance(match, str):
        html = match
        tag = None
    else:
        html = match.group(1)
        try:
            tag = match.group(2)
        except IndexError:
            tag = None
    tag = tag or re.match('.*?<(\\S).*?>', html).group(1)
    if raw and self.safe_mode:
        html = self._sanitize_html(html)
    elif 'markdown-in-html' in self.extras and 'markdown=' in html:
        first_line = html.split('\n', 1)[0]
        m = self._html_markdown_attr_re.search(first_line)
        if m:
            lines = html.split('\n')
            lines = list(filter(None, re.split('(.*?<%s.*markdown=.*?>)' % tag, lines[0]))) + lines[1:]
            lines = lines[:-1] + list(filter(None, re.split('(\\s*?</%s>.*?$)' % tag, lines[-1])))
            first_line = lines[0]
            middle = '\n'.join(lines[1:-1])
            last_line = lines[-1]
            first_line = first_line[:m.start()] + first_line[m.end():]
            f_key = _hash_text(first_line)
            self.html_blocks[f_key] = first_line
            l_key = _hash_text(last_line)
            self.html_blocks[l_key] = last_line
            return ''.join(['\n\n', f_key, '\n\n', middle, '\n\n', l_key, '\n\n'])
    elif self.extras.get('header-ids', {}).get('mixed') and self._h_tag_re.match(html):
        html = self._h_tag_re.sub(self._h_tag_sub, html)
    key = _hash_text(html)
    self.html_blocks[key] = html
    return '\n\n' + key + '\n\n'