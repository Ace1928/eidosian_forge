import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _updateParametersFromRunKwargs(self, **kwargs):
    """
        Updates attached params from __call__ without causing additional function runs
        """
    wasDisconnected = self.disconnect()
    try:
        for kwarg in set(kwargs).intersection(self.parameters):
            self.parameters[kwarg].setValue(kwargs[kwarg])
    finally:
        if not wasDisconnected:
            self.reconnect()
    for extraKey in set(kwargs) & set(self.extra):
        self.extra[extraKey] = kwargs[extraKey]