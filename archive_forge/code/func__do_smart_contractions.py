import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_smart_contractions(self, text):
    text = self._apostrophe_year_re.sub('&#8217;\\1', text)
    for c in self._contractions:
        text = text.replace("'%s" % c, '&#8217;%s' % c)
        text = text.replace("'%s" % c.capitalize(), '&#8217;%s' % c.capitalize())
    return text