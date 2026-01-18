import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_wavedrom_blocks(self, text):
    return self._fenced_code_block_re.sub(self._wavedrom_block_sub, text)