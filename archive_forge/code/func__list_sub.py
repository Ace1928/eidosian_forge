import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _list_sub(self, match):
    lst = match.group(1)
    lst_type = match.group(4) in self._marker_ul_chars and 'ul' or 'ol'
    if lst_type == 'ol' and match.group(4) != '1.':
        lst_opts = ' start="%s"' % match.group(4)[:-1]
    else:
        lst_opts = ''
    lst_opts = lst_opts + self._html_class_str_from_tag(lst_type)
    result = self._process_list_items(lst)
    if self.list_level:
        return '<%s%s>\n%s</%s>\n' % (lst_type, lst_opts, result, lst_type)
    else:
        return '<%s%s>\n%s</%s>\n\n' % (lst_type, lst_opts, result, lst_type)