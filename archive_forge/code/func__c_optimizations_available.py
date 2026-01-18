import os
import sys
def _c_optimizations_available():
    """
    Return the C optimization module, if available, otherwise
    a false value.

    If the optimizations are required but not available, this
    raises the ImportError.

    This does not say whether they should be used or not.
    """
    catch = () if _c_optimizations_required() else (ImportError,)
    try:
        from zope.interface import _zope_interface_coptimizations as c_opt
        return c_opt
    except catch:
        return False