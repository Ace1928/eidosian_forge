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
class StudyResult:
    """A collection of the input :class:`~.TuneDataset` and the tuning result

    :param dataset: input dataset for tuning
    :param result: tuning result as a dataframe

    .. attention::

        Do not construct this class directly.
    """

    def __init__(self, dataset: TuneDataset, result: WorkflowDataFrame):
        self._dataset = dataset
        self._result = result.persist().partition_by(TUNE_REPORT_ID, presort=TUNE_REPORT_METRIC).take(1).persist()

    def result(self, best_n: int=0) -> WorkflowDataFrame:
        """Get the top n results sorted by |SortMetric|

        :param best_n: number of result to get, defaults to 0.
          if `<=0` then it will return the entire result
        :return: result subset
        """
        if best_n <= 0:
            return self._result
        if len(self._dataset.keys) == 0:
            return self._result.take(n=best_n, presort=TUNE_REPORT_METRIC)
        else:
            return self._result.partition(by=self._dataset.keys, presort=TUNE_REPORT_METRIC).take(best_n)

    def next_tune_dataset(self, best_n: int=0) -> TuneDataset:
        """Convert the result back to a new :class:`~.TuneDataset` to be
        used by the next steps.

        :param best_n: top n result to extract, defaults to 0 (entire result)
        :return: a new dataset for tuning
        """
        data = self.result(best_n).drop([TUNE_REPORT_ID, TUNE_REPORT_METRIC, TUNE_REPORT], if_exists=True)
        return TuneDataset(data, dfs=self._dataset.dfs, keys=self._dataset.keys)

    def union_with(self, other: 'StudyResult') -> None:
        """Union with another result set and update itself

        :param other: the other result dataset

        .. note::
            This method also removes duplicated reports based on
            :meth:`tune.concepts.flow.trial.Trial.trial_id`. Each
            trial will have only the best report in the updated
            result
        """
        self._result = self._result.union(other._result).partition_by(TUNE_REPORT_ID, presort=TUNE_REPORT_METRIC).take(1).persist()