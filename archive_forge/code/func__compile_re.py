import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
@classmethod
def _compile_re(cls, s, flags):
    return re.compile(s.encode('ascii'), flags)