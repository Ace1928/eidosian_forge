import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='a Dask DataFrame', params='dask_obj : dask.dataframe.DataFrame', method='modin.core.execution.dask.implementations.pandas_on_dask.io.PandasOnDaskIO.from_dask')
def _from_dask(cls, dask_obj):
    return cls.io_cls.from_dask(dask_obj)