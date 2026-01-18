import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
class UnicodeWithAttrs(str):
    """A subclass of unicode used for the return value of conversion to
    possibly attach some attributes. E.g. the "toc_html" attribute when
    the "toc" extra is used.
    """
    metadata = None
    toc_html = None