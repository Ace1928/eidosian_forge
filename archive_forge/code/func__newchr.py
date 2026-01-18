from numbers import Integral
import string
import copy
from future.utils import istext, isbytes, PY2, PY3, with_metaclass
from future.types import no, issubset
from future.types.newobject import newobject
def _newchr(x):
    if isinstance(x, str):
        return x.encode('ascii')
    else:
        return chr(x)