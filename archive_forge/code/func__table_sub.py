import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _table_sub(self, match):
    trim_space_re = '^[ \t\n]+|[ \t\n]+$'
    trim_bar_re = '^\\||\\|$'
    split_bar_re = '^\\||(?<![\\`\\\\])\\|'
    escape_bar_re = '\\\\\\|'
    head, underline, body = match.groups()
    cols = [re.sub(escape_bar_re, '|', cell.strip()) for cell in re.split(split_bar_re, re.sub(trim_bar_re, '', re.sub(trim_space_re, '', underline)))]
    align_from_col_idx = {}
    for col_idx, col in enumerate(cols):
        if col[0] == ':' and col[-1] == ':':
            align_from_col_idx[col_idx] = ' style="text-align:center;"'
        elif col[0] == ':':
            align_from_col_idx[col_idx] = ' style="text-align:left;"'
        elif col[-1] == ':':
            align_from_col_idx[col_idx] = ' style="text-align:right;"'
    hlines = ['<table%s>' % self._html_class_str_from_tag('table'), '<thead%s>' % self._html_class_str_from_tag('thead'), '<tr>']
    cols = [re.sub(escape_bar_re, '|', cell.strip()) for cell in re.split(split_bar_re, re.sub(trim_bar_re, '', re.sub(trim_space_re, '', head)))]
    for col_idx, col in enumerate(cols):
        hlines.append('  <th%s>%s</th>' % (align_from_col_idx.get(col_idx, ''), self._run_span_gamut(col)))
    hlines.append('</tr>')
    hlines.append('</thead>')
    hlines.append('<tbody>')
    for line in body.strip('\n').split('\n'):
        hlines.append('<tr>')
        cols = [re.sub(escape_bar_re, '|', cell.strip()) for cell in re.split(split_bar_re, re.sub(trim_bar_re, '', re.sub(trim_space_re, '', line)))]
        for col_idx, col in enumerate(cols):
            hlines.append('  <td%s>%s</td>' % (align_from_col_idx.get(col_idx, ''), self._run_span_gamut(col)))
        hlines.append('</tr>')
    hlines.append('</tbody>')
    hlines.append('</table>')
    return '\n'.join(hlines) + '\n'