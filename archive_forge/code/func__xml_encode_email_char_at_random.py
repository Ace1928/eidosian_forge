import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _xml_encode_email_char_at_random(ch):
    r = random()
    if r > 0.9 and ch not in '@_':
        return ch
    elif r < 0.45:
        return '&#%s;' % hex(ord(ch))[1:]
    else:
        return '&#%s;' % ord(ch)