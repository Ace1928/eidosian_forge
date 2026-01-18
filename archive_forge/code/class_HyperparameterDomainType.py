from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class HyperparameterDomainType(enum.Enum):
    """Describes how to represent the set of known values for a hyperparameter."""
    INTERVAL = 'interval'
    DISCRETE_FLOAT = 'discrete_float'
    DISCRETE_STRING = 'discrete_string'
    DISCRETE_BOOL = 'discrete_bool'