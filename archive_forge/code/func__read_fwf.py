import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='a table of fixed-width formatted lines', params=_doc_io_method_kwargs_params, method='read_fwf')
def _read_fwf(cls, **kwargs):
    return cls.io_cls.read_fwf(**kwargs)