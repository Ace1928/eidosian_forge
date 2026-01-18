from typing import TYPE_CHECKING
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
def _transform_batch(self, df: 'DataBatchType') -> 'DataBatchType':
    for preprocessor in self.preprocessors:
        df = preprocessor.transform_batch(df)
    return df