import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _is_code_span(index, token):
    try:
        if token == '<code>':
            peek_tokens = split_tokens[index:index + 3]
        elif token == '</code>':
            peek_tokens = split_tokens[index - 2:index + 1]
        else:
            return False
    except IndexError:
        return False
    return re.match('<code>md5-[A-Fa-f0-9]{32}</code>', ''.join(peek_tokens))