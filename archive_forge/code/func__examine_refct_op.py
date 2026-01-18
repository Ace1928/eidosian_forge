import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
def _examine_refct_op(bb_lines):
    for num, ln in enumerate(bb_lines):
        m = _regex_incref.match(ln)
        if m is not None:
            yield (num, m.group(1), None)
            continue
        m = _regex_decref.match(ln)
        if m is not None:
            yield (num, None, m.group(1))
            continue
        yield (ln, None, None)