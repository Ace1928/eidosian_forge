import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='a SQL query', params=_doc_io_method_kwargs_params, method='read_sql_query')
def _read_sql_query(cls, **kwargs):
    return cls.io_cls.read_sql_query(**kwargs)