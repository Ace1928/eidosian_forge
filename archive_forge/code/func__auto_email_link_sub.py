import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _auto_email_link_sub(self, match):
    return self._encode_email_address(self._unescape_special_chars(match.group(1)))