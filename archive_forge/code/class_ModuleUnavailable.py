from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class ModuleUnavailable(object):
    """Mock object that raises :py:class:`.DeferredImportError` upon attribute access

    This object is returned by :py:func:`attempt_import()` in lieu of
    the module in the case that the module import fails.  Any attempts
    to access attributes on this object will raise a :py:class:`.DeferredImportError`
    exception.

    Parameters
    ----------
    name: str
        The module name that was being imported

    message: str
        The string message to return in the raised exception

    version_error: str
        A string to add to the message if the module failed to import because
        it did not match the required version

    import_error: str
        A string to add to the message documenting the Exception
        raised when the module failed to import.

    package: str
        The module name that originally attempted the import
    """
    _getattr_raises_attributeerror = {'__sphinx_mock__', '_dill'}

    def __init__(self, name, message, version_error, import_error, package):
        self.__name__ = name
        self._moduleunavailable_info_ = (message, version_error, import_error, package)

    def __getattr__(self, attr):
        if attr in ModuleUnavailable._getattr_raises_attributeerror:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, attr))
        raise DeferredImportError(self._moduleunavailable_message())

    def __getstate__(self):
        return (self.__name__, self._moduleunavailable_info_)

    def __setstate__(self, state):
        self.__name__, self._moduleunavailable_info_ = state

    def mro(self):
        return [ModuleUnavailable, object]

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

    def log_import_warning(self, logger='pyomo', msg=None):
        """Log the import error message to the specified logger

        This will log the the import error message to the specified
        logger.  If ``msg=`` is specified, it will override the default
        message passed to this instance of
        :py:class:`ModuleUnavailable`.

        """
        logging.getLogger(logger).warning(self._moduleunavailable_message(msg))

    @deprecated('use :py:class:`log_import_warning()`', version='6.0')
    def generate_import_warning(self, logger='pyomo.common'):
        self.log_import_warning(logger)