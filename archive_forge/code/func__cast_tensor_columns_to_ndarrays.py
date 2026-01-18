from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _cast_tensor_columns_to_ndarrays(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Cast all tensor extension columns in df to NumPy ndarrays."""
    pd = _lazy_import_pandas()
    try:
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    except AttributeError:
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    from ray.air.util.tensor_extensions.pandas import TensorDtype
    for col_name, col in df.items():
        if isinstance(col.dtype, TensorDtype):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                warnings.simplefilter('ignore', category=SettingWithCopyWarning)
                df.loc[:, col_name] = pd.Series(list(col.to_numpy()))
    return df