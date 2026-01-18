import functools
import re
import os
class UncaughtAttributeError(Exception):
    """
    Important, because `__getattr__` and `hasattr` catch AttributeErrors
    implicitly. This is really evil (mainly because of `__getattr__`).
    Therefore this class originally had to be derived from `BaseException`
    instead of `Exception`.  But because I removed relevant `hasattr` from
    the code base, we can now switch back to `Exception`.

    :param base: return values of sys.exc_info().
    """