from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _cast_ndarray_columns_to_tensor_extension(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Cast all NumPy ndarray columns in df to our tensor extension type, TensorArray.
    """
    pd = _lazy_import_pandas()
    try:
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    except AttributeError:
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    from ray.air.util.tensor_extensions.pandas import TensorArray, column_needs_tensor_extension
    for col_name, col in df.items():
        if column_needs_tensor_extension(col):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=FutureWarning)
                    warnings.simplefilter('ignore', category=SettingWithCopyWarning)
                    df.loc[:, col_name] = TensorArray(col)
            except Exception as e:
                raise ValueError(f'Tried to cast column {col_name} to the TensorArray tensor extension type but the conversion failed. To disable automatic casting to this tensor extension, set ctx = DataContext.get_current(); ctx.enable_tensor_extension_casting = False.') from e
    return df