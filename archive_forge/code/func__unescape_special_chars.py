import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _unescape_special_chars(self, text):
    hashmap = tuple(self._escape_table.items()) + tuple(self._code_table.items())
    hashmap += tuple((tuple(reversed(i)) for i in self.html_blocks.items()))
    while True:
        orig_text = text
        for ch, hash in hashmap:
            text = text.replace(hash, ch)
        if text == orig_text:
            break
    return text