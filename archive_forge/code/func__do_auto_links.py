import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_auto_links(self, text):
    text = self._auto_link_re.sub(self._auto_link_sub, text)
    text = self._auto_email_link_re.sub(self._auto_email_link_sub, text)
    return text