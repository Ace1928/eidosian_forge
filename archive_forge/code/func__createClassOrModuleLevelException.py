import sys
from . import case
from . import util
def _createClassOrModuleLevelException(self, result, exc, method_name, parent, info=None):
    errorName = f'{method_name} ({parent})'
    self._addClassOrModuleLevelException(result, exc, errorName, info)