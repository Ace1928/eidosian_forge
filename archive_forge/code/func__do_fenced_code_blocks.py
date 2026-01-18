import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_fenced_code_blocks(self, text):
    """Process ```-fenced unindented code blocks ('fenced-code-blocks' extra)."""
    return self._fenced_code_block_re.sub(self._fenced_code_block_sub, text)