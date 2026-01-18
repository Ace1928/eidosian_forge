from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class HyperparameterSort:
    """A sort criterium based on hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter to use for
        the sort. It corresponds to the hyperparameter_name field in the
        Hyperparameter class.
      sort_direction: The direction to sort.
    """
    hyperparameter_name: str
    sort_direction: HyperparameterSortDirection