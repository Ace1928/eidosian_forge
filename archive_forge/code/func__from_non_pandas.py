import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='a non-pandas object (dict, list, np.array etc...)', params=_doc_io_method_all_params, method='utils.from_non_pandas')
def _from_non_pandas(cls, *args, **kwargs):
    return cls.io_cls.from_non_pandas(*args, **kwargs)