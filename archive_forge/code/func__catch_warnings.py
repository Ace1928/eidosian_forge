import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
@contextlib.contextmanager
def _catch_warnings(self):
    """
        Replacement for warnings.catch_warnings.

        This method wraps warnings.catch_warnings, takes care to
        reset the warning registry before entering the with context,
        and ensures that DeprecationWarnings are always emitted.

        The hack to reset the warning registry is no longer needed in
        Python 3.4 and later. See http://bugs.python.org/issue4180 for
        more background.

        .. deprecated:: 6.2
            Use :func:`warnings.catch_warnings` instead.

        """
    warnings.warn('The _catch_warnings method is deprecated. Use warnings.catch_warnings instead.', DeprecationWarning)
    registry = sys._getframe(4).f_globals.get('__warningregistry__')
    if registry:
        registry.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        yield w