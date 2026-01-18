import re
import numpy as np
import pandas
import pyarrow
from modin.config import DoUseCalcite
from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar
from ..calcite_builder import CalciteBuilder
from ..calcite_serializer import CalciteSerializer
from ..dataframe.utils import ColNameCodec, is_supported_arrow_type
from ..db_worker import DbTable, DbWorker
from ..partitioning.partition import HdkOnNativeDataframePartition
@classmethod
def _get_unsupported_cols(cls, obj):
    """
        Return a list of columns with unsupported by HDK data types.

        Parameters
        ----------
        obj : pandas.DataFrame or pyarrow.Table
            Object to inspect on unsupported column types.

        Returns
        -------
        list
            List of unsupported columns.
        """
    if isinstance(obj, (pandas.Series, pandas.DataFrame)):
        if obj.empty:
            unsupported_cols = []
        elif isinstance(obj.columns, pandas.MultiIndex):
            unsupported_cols = [str(c) for c in obj.columns]
        else:
            cols = [name for name, col in obj.dtypes.items() if col == 'object']
            type_samples = obj.iloc[0][cols]
            unsupported_cols = [name for name, col in type_samples.items() if not isinstance(col, str) and (not (is_scalar(col) and pandas.isna(col)))]
        if len(unsupported_cols) > 0:
            return unsupported_cols
        try:
            schema = pyarrow.Schema.from_pandas(obj, preserve_index=False)
        except (pyarrow.lib.ArrowTypeError, pyarrow.lib.ArrowInvalid, ValueError, TypeError) as err:
            if type(err) is TypeError:
                if any([isinstance(t, pandas.SparseDtype) for t in obj.dtypes]):
                    ErrorMessage.single_warning('Sparse data is not currently supported!')
                else:
                    raise err
            if type(err) is ValueError and obj.columns.is_unique:
                raise err
            regex = 'Conversion failed for column ([^\\W]*)'
            unsupported_cols = []
            for msg in err.args:
                match = re.findall(regex, msg)
                unsupported_cols.extend(match)
            if len(unsupported_cols) == 0:
                unsupported_cols = obj.columns.tolist()
            return unsupported_cols
    else:
        schema = obj.schema
    return [field.name for field in schema if not is_supported_arrow_type(field.type)]