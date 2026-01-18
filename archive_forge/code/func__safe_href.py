import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
@property
def _safe_href(self):
    """
        _safe_href is adapted from pagedown's Markdown.Sanitizer.js
        From: https://github.com/StackExchange/pagedown/blob/master/LICENSE.txt
        Original Showdown code copyright (c) 2007 John Fraser
        Modifications and bugfixes (c) 2009 Dana Robinson
        Modifications and bugfixes (c) 2009-2014 Stack Exchange Inc.
        """
    safe = '-\\w'
    less_safe = '#/\\.!#$%&\\(\\)\\+,/:;=\\?@\\[\\]^`\\{\\}\\|~'
    domain = '(?:[%s]+(?:\\.[%s]+)*)(?:(?<!tel):\\d+/?)?(?![^:/]*:/*)' % (safe, safe)
    fragment = '[%s]*' % (safe + less_safe)
    return re.compile('^(?:(%s)?(%s)(%s)|(#|\\.{,2}/)(%s))$' % (self._safe_protocols, domain, fragment, fragment), re.I)