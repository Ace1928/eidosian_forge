import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _extract_link_def_sub(self, match):
    id, url, title = match.groups()
    key = id.lower()
    self.urls[key] = self._encode_amps_and_angles(url)
    if title:
        self.titles[key] = title
    return ''