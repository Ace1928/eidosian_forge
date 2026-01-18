from __future__ import annotations
import os
import warnings
from typing import IO, TYPE_CHECKING, Any, Hashable, Iterable, Optional, Union
import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import ArrayLike, Axis, DtypeObj, IndexKeyFunc, Scalar, Sequence
from pandas.api.types import is_integer
from pandas.core.arrays import ExtensionArray
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_dict_like, is_list_like
from pandas.core.series import _coerce_method
from pandas.io.formats.info import SeriesInfo
from pandas.util._validators import validate_bool_kwarg
from modin.config import PersistentPickle
from modin.logging import disable_logging
from modin.pandas.io import from_pandas, to_pandas
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from .accessor import CachedAccessor, SparseAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .iterator import PartitionIterator
from .series_utils import (
from .utils import _doc_binary_op, cast_function_modin2pandas, is_scalar
@classmethod
def _inflate_light(cls, query_compiler, name, source_pid) -> Series:
    """
        Re-creates the object from previously-serialized lightweight representation.

        The method is used for faster but not disk-storable persistence.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to use for object re-creation.
        name : str
            The name to give to the new object.
        source_pid : int
            Determines whether a Modin or pandas object needs to be created.
            Modin objects are created only on the main process.

        Returns
        -------
        Series
            New Series based on the `query_compiler`.
        """
    if os.getpid() != source_pid:
        res = query_compiler.to_pandas()
        if res.columns == [MODIN_UNNAMED_SERIES_LABEL]:
            res = res.squeeze(axis=1)
            res.name = None
        return res
    return cls(query_compiler=query_compiler, name=name)