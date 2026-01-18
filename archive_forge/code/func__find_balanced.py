import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _find_balanced(self, text, start, open_c, close_c):
    """Returns the index where the open_c and close_c characters balance
        out - the same number of open_c and close_c are encountered - or the
        end of string if it's reached before the balance point is found.
        """
    i = start
    l = len(text)
    count = 1
    while count > 0 and i < l:
        if text[i] == open_c:
            count += 1
        elif text[i] == close_c:
            count -= 1
        i += 1
    return i