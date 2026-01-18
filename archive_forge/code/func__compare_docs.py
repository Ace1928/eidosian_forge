import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
def _compare_docs(old_func, new_func):
    old_doc = inspect.getdoc(old_func)
    new_doc = inspect.getdoc(new_func)
    index = new_doc.index('\n\n') + 2
    assert_equal(new_doc[index:], old_doc)