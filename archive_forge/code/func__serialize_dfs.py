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
def _serialize_dfs(self) -> Tuple[WorkflowDataFrame, List[str]]:
    df = self._serialize_df(self._dfs_spec[0][1], self._dfs_spec[0][0])
    keys = list(self._dfs_spec[0][1].partition_spec.partition_by)
    for i in range(1, len(self._dfs_spec)):
        how = self._dfs_spec[i][2]
        new_keys = set(self._dfs_spec[i][1].partition_spec.partition_by)
        next_df = self._serialize_df(self._dfs_spec[i][1], self._dfs_spec[i][0])
        df = df.join(next_df, how=how)
        if how != 'cross':
            keys = [k for k in keys if k in new_keys]
    return (df, keys)