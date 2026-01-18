import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _find_non_whitespace(self, text, start):
    """Returns the index of the first non-whitespace character in text
        after (and including) start
        """
    match = self._whitespace.match(text, start)
    return match.end()