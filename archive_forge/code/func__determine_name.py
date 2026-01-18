from __future__ import annotations
import warnings
from typing import Hashable, Iterable, Mapping, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import ArrayLike, DtypeBackend, Scalar, npt
from pandas.core.dtypes.common import is_list_like
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from modin.pandas.io import to_pandas
from modin.utils import _inherit_docstrings
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
def _determine_name(objs: Iterable[BaseQueryCompiler], axis: Union[int, str]):
    """
    Determine names of index after concatenation along passed axis.

    Parameters
    ----------
    objs : iterable of QueryCompilers
        Objects to concatenate.
    axis : int or str
        The axis to concatenate along.

    Returns
    -------
    list with single element
        Computed index name, `None` if it could not be determined.
    """
    axis = pandas.DataFrame()._get_axis_number(axis)

    def get_names(obj):
        return obj.columns.names if axis else obj.index.names
    names = np.array([get_names(obj) for obj in objs])
    if np.all(names == names[0]):
        return list(names[0]) if is_list_like(names[0]) else [names[0]]
    else:
        return None