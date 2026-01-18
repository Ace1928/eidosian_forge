from __future__ import annotations
import typing
from copy import copy, deepcopy
from typing import Iterable, List, cast, overload
import pandas as pd
from ._utils import array_kind, check_required_aesthetics, ninteraction
from .exceptions import PlotnineError
from .mapping.aes import NO_GROUP, SCALED_AESTHETICS, aes
from .mapping.evaluation import evaluate, stage
def _make_layer_data(self, plot_data: DataLike | None):
    """
        Generate data to be used by this layer

        Parameters
        ----------
        plot_data :
            ggplot object data
        """
    if plot_data is None:
        data = pd.DataFrame()
    elif hasattr(plot_data, 'to_pandas'):
        data = cast('DataFrameConvertible', plot_data).to_pandas()
    else:
        data = cast('pd.DataFrame', plot_data)
    if self._data is None:
        try:
            self.data = copy(data)
        except AttributeError as e:
            _geom_name = self.geom.__class__.__name__
            _data_name = data.__class__.__name__
            msg = f'{_geom_name} layer expects a dataframe, but it got {_data_name} instead.'
            raise PlotnineError(msg) from e
    elif callable(self._data):
        self.data = self._data(data)
        if not isinstance(self.data, pd.DataFrame):
            raise PlotnineError('Data function must return a Pandas dataframe')
    elif hasattr(self._data, 'to_pandas'):
        self.data = cast('DataFrameConvertible', self._data).to_pandas()
    elif isinstance(self._data, pd.DataFrame):
        self.data = self._data.copy()
    else:
        raise TypeError(f'Data has a bad type: {type(self.data)}')