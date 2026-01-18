import os
import sys
def _should_attempt_c_optimizations():
    """
    Return a true value if we should attempt to use the C optimizations.

    This takes into account whether we're on PyPy and the value of the
    ``PURE_PYTHON`` environment variable, as defined in `_use_c_impl`.
    """
    is_pypy = hasattr(sys, 'pypy_version_info')
    if _c_optimizations_required():
        return True
    if is_pypy:
        return False
    return not _c_optimizations_ignored()