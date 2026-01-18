import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_tg_spoiler(self, text):
    text = self._tg_spoiler_re.sub('<tg-spoiler>\\1</tg-spoiler>', text)
    return text