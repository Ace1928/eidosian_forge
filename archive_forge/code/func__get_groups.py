from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
import pandas
import pandas.core.resample
from pandas._libs import lib
from pandas.core.dtypes.common import is_list_like
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings
def _get_groups(self):
    """
        Compute the resampled groups.

        Returns
        -------
        PandasGroupby
            Groups as specified by resampling arguments.
        """
    df = self._dataframe if self.axis == 0 else self._dataframe.T
    convention = self.resample_kwargs['convention']
    groups = df.groupby(pandas.Grouper(key=self.resample_kwargs['on'], freq=self.resample_kwargs['rule'], closed=self.resample_kwargs['closed'], label=self.resample_kwargs['label'], convention=convention if convention is not lib.no_default else 'start', level=self.resample_kwargs['level'], origin=self.resample_kwargs['origin'], offset=self.resample_kwargs['offset']), group_keys=self.resample_kwargs['group_keys'])
    return groups