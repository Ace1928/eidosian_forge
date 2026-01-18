import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _perfcheck(object=None):
    import time
    if object is None:
        object = [('string', (1, 2), [3, 4], {5: 6, 7: 8})] * 100000
    p = PrettyPrinter()
    t1 = time.perf_counter()
    p._safe_repr(object, {}, None, 0, True)
    t2 = time.perf_counter()
    p.pformat(object)
    t3 = time.perf_counter()
    print('_safe_repr:', t2 - t1)
    print('pformat:', t3 - t2)