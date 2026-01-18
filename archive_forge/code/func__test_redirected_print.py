import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
def _test_redirected_print(x, tp, ref=None):
    file = StringIO()
    file_tp = StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = file_tp
        print(tp(x))
        sys.stdout = file
        if ref:
            print(ref)
        else:
            print(x)
    finally:
        sys.stdout = stdout
    assert_equal(file.getvalue(), file_tp.getvalue(), err_msg='print failed for type%s' % tp)