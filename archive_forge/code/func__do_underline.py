import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_underline(self, text):
    text = self._underline_re.sub('<u>\\1</u>', text)
    return text