import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_lists(self, text):
    pos = 0
    while True:
        hits = []
        for marker_pat in (self._marker_ul, self._marker_ol):
            less_than_tab = self.tab_width - 1
            other_marker_pat = self._marker_ul if marker_pat == self._marker_ol else self._marker_ol
            whole_list = "\n                    (                   # \\1 = whole list\n                      (                 # \\2\n                        ([ ]{0,%d})     # \\3 = the indentation level of the list item marker\n                        (%s)            # \\4 = first list item marker\n                        [ \\t]+\n                        (?!\\ *\\4\\ )     # '- - - ...' isn't a list. See 'not_quite_a_list' test case.\n                      )\n                      (?:.+?)\n                      (                 # \\5\n                          \\Z\n                        |\n                          \\n{2,}\n                          (?=\\S)\n                          (?!           # Negative lookahead for another list item marker\n                            [ \\t]*\n                            %s[ \\t]+\n                          )\n                        |\n                          \\n+\n                          (?=\n                            \\3          # lookahead for a different style of list item marker\n                            %s[ \\t]+\n                          )\n                      )\n                    )\n                " % (less_than_tab, marker_pat, marker_pat, other_marker_pat)
            if self.list_level:
                list_re = re.compile('^' + whole_list, re.X | re.M | re.S)
            else:
                list_re = re.compile('(?:(?<=\\n\\n)|\\A\\n?)' + whole_list, re.X | re.M | re.S)
            match = list_re.search(text, pos)
            if match:
                hits.append((match.start(), match))
        if not hits:
            break
        hits.sort()
        match = hits[0][1]
        start, end = match.span()
        middle = self._list_sub(match)
        text = text[:start] + middle + text[end:]
        pos = start + len(middle)
    return text