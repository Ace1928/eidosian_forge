import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
def _set_opt_string(self, opts):
    if self.REMAINING in opts:
        self._long_opts.append(self.REMAINING)
    optparse.Option._set_opt_string(self, opts)
    if len(self._short_opts) > len(self._long_opts):
        raise OptionError('goption.Option needs more long option names than short option names')