from __future__ import absolute_import, print_function, division
import os
import sys
import pytest
from petl.compat import izip_longest
def _ieq_row(re, ra, ir):
    assert ra is not None, 'Expected row #%d is None, but result row is not None' % ir
    assert re is not None, 'Expected row #%d is not None, but result row is None' % ir
    ic = 0
    for ve, va in izip_longest(re, ra, fillvalue=None):
        if isinstance(ve, list):
            for je, ja in izip_longest(ve, va, fillvalue=None):
                _ieq_col(je, ja, re, ra, ir, ic)
        elif not isinstance(ve, dict):
            _ieq_col(ve, va, re, ra, ir, ic)
        ic = ic + 1