import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
import cloudpickle
import numpy as np
import pandas as pd
from fugue import (
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
from tune.exceptions import TuneCompileError
def add_df(self, name: str, df: WorkflowDataFrame, how: str='') -> 'TuneDatasetBuilder':
    """Add a dataframe to the dataset

        :param name: name of the dataframe, it will also create a
          ``__tune_df__<name>`` column in the dataset dataframe
        :param df: the dataframe to add.
        :param how: join type, can accept ``semi``, ``left_semi``,
          ``anti``, ``left_anti``, ``inner``, ``left_outer``,
          ``right_outer``, ``full_outer``, ``cross``
        :returns: the builder itself

        .. note::

            For the first dataframe you add, ``how`` should be empty.
            From the second dataframe you add, ``how`` must be set.

        .. note::

            If ``df`` is prepartitioned, the partition key will be used to
            join with the added dataframes. Read
            :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
            for more details
        """
    assert_or_throw(not any((r[0] == name for r in self._dfs_spec)), TuneCompileError(name + ' already exists'))
    if len(self._dfs_spec) == 0:
        assert_or_throw(how == '', TuneCompileError("first dataframe can't specify how to join"))
    else:
        assert_or_throw(how != '', TuneCompileError('must specify how to join after first dataframe'))
    self._dfs_spec.append((name, df, how))
    return self