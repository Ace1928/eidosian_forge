import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _encode_incomplete_tags(self, text):
    if self.safe_mode not in ('replace', 'escape'):
        return text
    if text.endswith('>'):
        return text

    def incomplete_tags_sub(match):
        return match.group().replace('<', '&lt;')
    return self._incomplete_tags_re.sub(incomplete_tags_sub, text)