from numbers import Integral
import string
from future.utils import istext, isbytes, PY2, with_metaclass
from future.types import no, issubset
class newmemoryview(object):
    """
    A pretty lame backport of the Python 2.7 and Python 3.x
    memoryviewview object to Py2.6.
    """

    def __init__(self, obj):
        return obj