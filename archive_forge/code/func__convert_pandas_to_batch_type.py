from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _convert_pandas_to_batch_type(data: 'pd.DataFrame', type: BatchFormat, cast_tensor_columns: bool=False) -> DataBatchType:
    """Convert the provided Pandas dataframe to the provided ``type``.

    Args:
        data: A Pandas DataFrame
        type: The specific ``BatchFormat`` to convert to.
        cast_tensor_columns: Whether tensor columns should be cast to our tensor
            extension type.

    Returns:
        The input data represented with the provided type.
    """
    if cast_tensor_columns:
        data = _cast_ndarray_columns_to_tensor_extension(data)
    if type == BatchFormat.PANDAS:
        return data
    elif type == BatchFormat.NUMPY:
        if len(data.columns) == 1:
            return data.iloc[:, 0].to_numpy()
        else:
            output_dict = {}
            for column in data:
                output_dict[column] = data[column].to_numpy()
            return output_dict
    elif type == BatchFormat.ARROW:
        if not pyarrow:
            raise ValueError('Attempted to convert data to Pyarrow Table but Pyarrow is not installed. Please do `pip install pyarrow` to install Pyarrow.')
        return pyarrow.Table.from_pandas(data)
    else:
        raise ValueError(f'Received type {type}, but expected it to be one of {DataBatchType}')