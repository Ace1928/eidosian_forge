from collections import Counter
from numbers import Number
from typing import Dict, List, Optional, Union
import pandas as pd
from pandas.api.types import is_categorical_dtype
from ray.data import Dataset
from ray.data.aggregate import Mean
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def _get_most_frequent_values(dataset: Dataset, *columns: str) -> Dict[str, Union[str, Number]]:
    columns = list(columns)

    def get_pd_value_counts(df: pd.DataFrame) -> List[Dict[str, Counter]]:
        return {col: [Counter(df[col].value_counts().to_dict())] for col in columns}
    value_counts = dataset.map_batches(get_pd_value_counts, batch_format='pandas')
    final_counters = {col: Counter() for col in columns}
    for batch in value_counts.iter_batches(batch_size=None):
        for col, counters in batch.items():
            for counter in counters:
                final_counters[col] += counter
    return {f'most_frequent({column})': final_counters[column].most_common(1)[0][0] for column in columns}