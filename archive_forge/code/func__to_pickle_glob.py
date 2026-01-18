import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_pickle_glob(cls, *args, **kwargs):
    """
        Distributed pickle query compiler object.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
    current_execution = get_current_execution()
    if current_execution not in supported_executions:
        raise NotImplementedError(f'`_to_pickle_glob()` is not implemented for {current_execution} execution.')
    return cls.io_cls.to_pickle_glob(*args, **kwargs)