from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _moduleunavailable_message(self, msg=None):
    _err, _ver, _imp, _package = self._moduleunavailable_info_
    if msg is None:
        msg = _err
    if _imp:
        if not msg or not str(msg):
            _pkg_str = _package.split('.')[0].capitalize()
            if _pkg_str:
                _pkg_str += ' '
            msg = 'The %s module (an optional %sdependency) failed to import: %s' % (self.__name__, _pkg_str, _imp)
        else:
            msg = '%s (import raised %s)' % (msg, _imp)
    if _ver:
        if not msg or not str(msg):
            msg = 'The %s module %s' % (self.__name__, _ver)
        else:
            msg = '%s (%s)' % (msg, _ver)
    return msg