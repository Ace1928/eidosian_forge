import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def _subDescription(self):
    parts = []
    if self._message is not _subtest_msg_sentinel:
        parts.append('[{}]'.format(self._message))
    if self.params:
        params_desc = ', '.join(('{}={!r}'.format(k, v) for k, v in self.params.items()))
        parts.append('({})'.format(params_desc))
    return ' '.join(parts) or '(<subtest>)'