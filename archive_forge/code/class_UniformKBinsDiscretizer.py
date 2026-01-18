from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class UniformKBinsDiscretizer(_AbstractKBinsDiscretizer):
    """Bin values into discrete intervals (bins) of uniform width.

    Columns must contain numerical values.

    Examples:
        Use :class:`UniformKBinsDiscretizer` to bin continuous features.

        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import UniformKBinsDiscretizer
        >>> df = pd.DataFrame({
        ...     "value_1": [0.2, 1.4, 2.5, 6.2, 9.7, 2.1],
        ...     "value_2": [10, 15, 13, 12, 23, 25],
        ... })
        >>> ds = ray.data.from_pandas(df)
        >>> discretizer = UniformKBinsDiscretizer(
        ...     columns=["value_1", "value_2"], bins=4
        ... )
        >>> discretizer.fit_transform(ds).to_pandas()
           value_1  value_2
        0        0        0
        1        0        1
        2        0        0
        3        2        0
        4        3        3
        5        0        3

        You can also specify different number of bins per column.

        >>> discretizer = UniformKBinsDiscretizer(
        ...     columns=["value_1", "value_2"], bins={"value_1": 4, "value_2": 3}
        ... )
        >>> discretizer.fit_transform(ds).to_pandas()
           value_1  value_2
        0        0        0
        1        0        0
        2        0        0
        3        2        0
        4        3        2
        5        0        2


    Args:
        columns: The columns to discretize.
        bins: Defines the number of equal-width bins.
            Can be either an integer (which will be applied to all columns),
            or a dict that maps columns to integers.
            The range is extended by .1% on each side to include
            the minimum and maximum values.
        right: Indicates whether bins includes the rightmost edge or not.
        include_lowest: Whether the first interval should be left-inclusive
            or not.
        duplicates: Can be either 'raise' or 'drop'. If bin edges are not unique,
            raise ``ValueError`` or drop non-uniques.
        dtypes: An optional dictionary that maps columns to ``pd.CategoricalDtype``
            objects or ``np.integer`` types. If you don't include a column in ``dtypes``
            or specify it as an integer dtype, the outputted column will consist of
            ordered integers corresponding to bins. If you use a
            ``pd.CategoricalDtype``, the outputted column will be a
            ``pd.CategoricalDtype`` with the categories being mapped to bins.
            You can use ``pd.CategoricalDtype(categories, ordered=True)`` to
            preserve information about bin order.

    .. seealso::

        :class:`CustomKBinsDiscretizer`
            If you want to specify your own bin edges.
    """

    def __init__(self, columns: List[str], bins: Union[int, Dict[str, int]], *, right: bool=True, include_lowest: bool=False, duplicates: str='raise', dtypes: Optional[Dict[str, Union[pd.CategoricalDtype, Type[np.integer]]]]=None):
        self.columns = columns
        self.bins = bins
        self.right = right
        self.include_lowest = include_lowest
        self.duplicates = duplicates
        self.dtypes = dtypes

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self._validate_on_fit()
        stats = {}
        aggregates = []
        if isinstance(self.bins, dict):
            columns = self.bins.keys()
        else:
            columns = self.columns
        for column in columns:
            aggregates.extend(self._fit_uniform_covert_bin_to_aggregate_if_needed(column))
        aggregate_stats = dataset.aggregate(*aggregates)
        mins = {}
        maxes = {}
        for key, value in aggregate_stats.items():
            column_name = key[4:-1]
            if key.startswith('min'):
                mins[column_name] = value
            if key.startswith('max'):
                maxes[column_name] = value
        for column in mins.keys():
            bins = self.bins[column] if isinstance(self.bins, dict) else self.bins
            stats[column] = _translate_min_max_number_of_bins_to_bin_edges(mins[column], maxes[column], bins, self.right)
        self.stats_ = stats
        return self

    def _validate_on_fit(self):
        self._validate_bins_columns()

    def _fit_uniform_covert_bin_to_aggregate_if_needed(self, column: str):
        bins = self.bins[column] if isinstance(self.bins, dict) else self.bins
        if isinstance(bins, int):
            return (Min(column), Max(column))
        else:
            raise TypeError(f'`bins` must be an integer or a dict of integers, got {bins}')