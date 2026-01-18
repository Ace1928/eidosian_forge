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
def _space_to_df(self, wf: FugueWorkflow, batch_size: int=1, shuffle: bool=True) -> WorkflowDataFrame:

    def get_data() -> Iterable[List[Any]]:
        it = list(self._space)
        if shuffle:
            random.seed(0)
            random.shuffle(it)
        res: List[Any] = []
        for a in it:
            res.append(a)
            if batch_size == len(res):
                yield [cloudpickle.dumps(res)]
                res = []
        if len(res) > 0:
            yield [cloudpickle.dumps(res)]
    return wf.df(IterableDataFrame(get_data(), f'{TUNE_DATASET_PARAMS_PREFIX}:binary'))