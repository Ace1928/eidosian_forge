import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
class TrainingCheckPoint(TrainingCallback):
    """Checkpointing operation.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    directory :
        Output model directory.
    name :
        pattern of output model file.  Models will be saved as name_0.json, name_1.json,
        name_2.json ....
    as_pickle :
        When set to True, all training parameters will be saved in pickle format, instead
        of saving only the model.
    iterations :
        Interval of checkpointing.  Checkpointing is slow so setting a larger number can
        reduce performance hit.

    """

    def __init__(self, directory: Union[str, os.PathLike], name: str='model', as_pickle: bool=False, iterations: int=100) -> None:
        self._path = os.fspath(directory)
        self._name = name
        self._as_pickle = as_pickle
        self._iterations = iterations
        self._epoch = 0
        super().__init__()

    def after_iteration(self, model: _Model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if self._epoch == self._iterations:
            path = os.path.join(self._path, self._name + '_' + str(epoch) + ('.pkl' if self._as_pickle else '.json'))
            self._epoch = 0
            if collective.get_rank() == 0:
                if self._as_pickle:
                    with open(path, 'wb') as fd:
                        pickle.dump(model, fd)
                else:
                    model.save_model(path)
        self._epoch += 1
        return False