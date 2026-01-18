import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _tag_is_closed(self, tag_name, text):
    return len(re.findall('<%s(?:.*?)>' % tag_name, text)) == len(re.findall('</%s>' % tag_name, text))