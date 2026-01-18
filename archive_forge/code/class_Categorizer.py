from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class Categorizer(Preprocessor):
    """Convert columns to ``pd.CategoricalDtype``.

    Use this preprocessor with frameworks that have built-in support for
    ``pd.CategoricalDtype`` like LightGBM.

    .. warning::

        If you don't specify ``dtypes``, fit this preprocessor before splitting
        your dataset into train and test splits. This ensures categories are
        consistent across splits.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import Categorizer
        >>>
        >>> df = pd.DataFrame(
        ... {
        ...     "sex": ["male", "female", "male", "female"],
        ...     "level": ["L4", "L5", "L3", "L4"],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> categorizer = Categorizer(columns=["sex", "level"])
        >>> categorizer.fit_transform(ds).schema().types  # doctest: +SKIP
        [CategoricalDtype(categories=['female', 'male'], ordered=False), CategoricalDtype(categories=['L3', 'L4', 'L5'], ordered=False)]

        If you know the categories in advance, you can specify the categories with the
        ``dtypes`` parameter.

        >>> categorizer = Categorizer(
        ...     columns=["sex", "level"],
        ...     dtypes={"level": pd.CategoricalDtype(["L3", "L4", "L5", "L6"], ordered=True)},
        ... )
        >>> categorizer.fit_transform(ds).schema().types  # doctest: +SKIP
        [CategoricalDtype(categories=['female', 'male'], ordered=False), CategoricalDtype(categories=['L3', 'L4', 'L5', 'L6'], ordered=True)]

    Args:
        columns: The columns to convert to ``pd.CategoricalDtype``.
        dtypes: An optional dictionary that maps columns to ``pd.CategoricalDtype``
            objects. If you don't include a column in ``dtypes``, the categories
            are inferred.
    """

    def __init__(self, columns: List[str], dtypes: Optional[Dict[str, pd.CategoricalDtype]]=None):
        if not dtypes:
            dtypes = {}
        self.columns = columns
        self.dtypes = dtypes

    def _fit(self, dataset: Dataset) -> Preprocessor:
        columns_to_get = [column for column in self.columns if column not in self.dtypes]
        if columns_to_get:
            unique_indices = _get_unique_value_indices(dataset, columns_to_get, drop_na_values=True, key_format='{0}')
            unique_indices = {column: pd.CategoricalDtype(values_indices.keys()) for column, values_indices in unique_indices.items()}
        else:
            unique_indices = {}
        unique_indices = {**self.dtypes, **unique_indices}
        self.stats_: Dict[str, pd.CategoricalDtype] = unique_indices
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        df = df.astype(self.stats_)
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r}, dtypes={self.dtypes!r})'