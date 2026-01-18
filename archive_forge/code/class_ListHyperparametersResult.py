from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class ListHyperparametersResult:
    """The result from calling list_hyperparameters().

    Attributes:
      hyperparameters: The hyperparameteres belonging to the experiments in the
        request.
      session_groups: The session groups present in the experiments in the
        request.
    """
    hyperparameters: Collection[Hyperparameter]
    session_groups: Collection[HyperparameterSessionGroup]