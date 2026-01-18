import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _is_auto_link(s):
    if ':' in s and self._auto_link_re.match(s):
        return True
    elif '@' in s and self._auto_email_link_re.match(s):
        return True
    return False