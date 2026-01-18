import sys
from . import case
from . import util
def _addClassOrModuleLevelException(self, result, exception, errorName, info=None):
    error = _ErrorHolder(errorName)
    addSkip = getattr(result, 'addSkip', None)
    if addSkip is not None and isinstance(exception, case.SkipTest):
        addSkip(error, str(exception))
    elif not info:
        result.addError(error, sys.exc_info())
    else:
        result.addError(error, info)