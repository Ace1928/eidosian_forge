import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _form_paragraphs(self, text):
    text = text.strip('\n')
    grafs = []
    for i, graf in enumerate(re.split('\\n{2,}', text)):
        if graf in self.html_blocks:
            grafs.append(self.html_blocks[graf])
        else:
            cuddled_list = None
            if 'cuddled-lists' in self.extras:
                li = self._list_item_re.search(graf + '\n')
                if li and len(li.group(2)) <= 3 and (li.group('next_marker') and li.group('marker')[-1] == li.group('next_marker')[-1] or li.group('next_marker') is None):
                    start = li.start()
                    cuddled_list = self._do_lists(graf[start:]).rstrip('\n')
                    assert re.match('^<(?:ul|ol).*?>', cuddled_list)
                    graf = graf[:start]
            graf = self._run_span_gamut(graf)
            grafs.append('<p%s>' % self._html_class_str_from_tag('p') + graf.lstrip(' \t') + '</p>')
            if cuddled_list:
                grafs.append(cuddled_list)
    return '\n\n'.join(grafs)