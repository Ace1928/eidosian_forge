import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
@classmethod
def _make_boundary(cls, text=None):
    token = random.randrange(sys.maxsize)
    boundary = '=' * 15 + _fmt % token + '=='
    if text is None:
        return boundary
    b = boundary
    counter = 0
    while True:
        cre = cls._compile_re('^--' + re.escape(b) + '(--)?$', re.MULTILINE)
        if not cre.search(text):
            break
        b = boundary + '.' + str(counter)
        counter += 1
    return b