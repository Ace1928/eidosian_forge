import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _regex_from_encoded_pattern(s):
    """'foo'    -> re.compile(re.escape('foo'))
       '/foo/'  -> re.compile('foo')
       '/foo/i' -> re.compile('foo', re.I)
    """
    if s.startswith('/') and s.rfind('/') != 0:
        idx = s.rfind('/')
        _, flags_str = (s[1:idx], s[idx + 1:])
        flag_from_char = {'i': re.IGNORECASE, 'l': re.LOCALE, 's': re.DOTALL, 'm': re.MULTILINE, 'u': re.UNICODE}
        flags = 0
        for char in flags_str:
            try:
                flags |= flag_from_char[char]
            except KeyError:
                raise ValueError("unsupported regex flag: '%s' in '%s' (must be one of '%s')" % (char, s, ''.join(list(flag_from_char.keys()))))
        return re.compile(s[1:idx], flags)
    else:
        return re.compile(re.escape(s))