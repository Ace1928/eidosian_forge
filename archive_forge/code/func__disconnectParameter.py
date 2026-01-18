import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _disconnectParameter(self, param):
    param.sigValueChanged.disconnect(self.updateCachedParameterValues)
    for signal in (param.sigValueChanging, param.sigValueChanged):
        fn.disconnect(signal, self.runFromChangedOrChanging)