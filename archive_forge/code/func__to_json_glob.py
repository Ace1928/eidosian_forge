import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_json_glob(cls, *args, **kwargs):
    """
        Write query compiler content to several json files.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
    current_execution = get_current_execution()
    if current_execution not in supported_executions:
        raise NotImplementedError(f'`_to_json_glob()` is not implemented for {current_execution} execution.')
    return cls.io_cls.to_json_glob(*args, **kwargs)