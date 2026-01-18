from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class MinMaxScaler(Preprocessor):
    """Scale each column by its range.

    The general formula is given by

    .. math::

        x' = \\frac{x - \\min(x)}{\\max{x} - \\min{x}}

    where :math:`x` is the column and :math:`x'` is the transformed column. If
    :math:`\\max{x} - \\min{x} = 0` (i.e., the column is constant-valued), then the
    transformed column will get filled with zeros.

    Transformed values are always in the range :math:`[0, 1]`.

    .. tip::
        This can be used as an alternative to :py:class:`StandardScaler`.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import MinMaxScaler
        >>>
        >>> df = pd.DataFrame({"X1": [-2, 0, 2], "X2": [-3, -3, 3], "X3": [1, 1, 1]})   # noqa: E501
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> ds.to_pandas()  # doctest: +SKIP
           X1  X2  X3
        0  -2  -3   1
        1   0  -3   1
        2   2   3   1

        Columns are scaled separately.

        >>> preprocessor = MinMaxScaler(columns=["X1", "X2"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
            X1   X2  X3
        0  0.0  0.0   1
        1  0.5  0.0   1
        2  1.0  1.0   1

        Constant-valued columns get filled with zeros.

        >>> preprocessor = MinMaxScaler(columns=["X3"])
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
           X1  X2   X3
        0  -2  -3  0.0
        1   0  -3  0.0
        2   2   3  0.0

    Args:
        columns: The columns to separately scale.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def _fit(self, dataset: Dataset) -> Preprocessor:
        aggregates = [Agg(col) for Agg in [Min, Max] for col in self.columns]
        self.stats_ = dataset.aggregate(*aggregates)
        return self

    def _transform_pandas(self, df: pd.DataFrame):

        def column_min_max_scaler(s: pd.Series):
            s_min = self.stats_[f'min({s.name})']
            s_max = self.stats_[f'max({s.name})']
            diff = s_max - s_min
            if diff == 0:
                diff = 1
            return (s - s_min) / diff
        df.loc[:, self.columns] = df.loc[:, self.columns].transform(column_min_max_scaler)
        return df

    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns!r})'