import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_link_patterns(self, text):
    link_from_hash = {}
    for regex, repl in self.link_patterns:
        replacements = []
        for match in regex.finditer(text):
            if any((self._match_overlaps_substr(text, match, h) for h in link_from_hash)):
                continue
            if hasattr(repl, '__call__'):
                href = repl(match)
            else:
                href = match.expand(repl)
            replacements.append((match.span(), href))
        for (start, end), href in reversed(replacements):
            if text[start - 1:start] == '[' and text[end:end + 1] == ']':
                continue
            if text[start - 2:start] == '](' or text[end:end + 2] == '")':
                continue
            if text[start - 3:start] == '"""' and text[end:end + 3] == '"""':
                text = text[:start - 3] + text[start:end] + text[end + 3:]
                continue
            is_inside_link = False
            for link_re in (self._auto_link_re, self._basic_link_re):
                for match in link_re.finditer(text):
                    if any((r[0] <= start and end <= r[1] for r in match.regs)):
                        is_inside_link = True
                        break
                else:
                    continue
                break
            if is_inside_link:
                continue
            escaped_href = href.replace('"', '&quot;').replace('*', self._escape_table['*']).replace('_', self._escape_table['_'])
            link = '<a href="%s">%s</a>' % (escaped_href, text[start:end])
            hash = _hash_text(link)
            link_from_hash[hash] = link
            text = text[:start] + hash + text[end:]
    for hash, link in list(link_from_hash.items()):
        text = text.replace(hash, link)
    return text