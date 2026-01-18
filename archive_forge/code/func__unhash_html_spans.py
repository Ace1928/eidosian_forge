import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _unhash_html_spans(self, text):
    for key, sanitized in list(self.html_spans.items()):
        text = text.replace(key, sanitized)
    return text