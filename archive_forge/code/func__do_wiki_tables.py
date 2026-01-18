import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_wiki_tables(self, text):
    if '||' not in text:
        return text
    less_than_tab = self.tab_width - 1
    wiki_table_re = re.compile('\n            (?:(?<=\\n\\n)|\\A\\n?)            # leading blank line\n            ^([ ]{0,%d})\\|\\|.+?\\|\\|[ ]*\\n  # first line\n            (^\\1\\|\\|.+?\\|\\|\\n)*        # any number of subsequent lines\n            ' % less_than_tab, re.M | re.X)
    return wiki_table_re.sub(self._wiki_table_sub, text)