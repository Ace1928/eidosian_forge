from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class HyperparameterValue:
    """A hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameters. It
        corresponds to the hyperparameter_name field in the Hyperparameter
        class.
      domain_type: A HyperparameterDomainType describing how we represent the
        set of known values in the `domain` attribute.
      value: The value of the hyperparameter.

        If domain_type is INTERVAL or DISCRETE_FLOAT, value is a float.
        If domain_type is DISCRETE_STRING, value is a str.
        If domain_type is DISCRETE_BOOL, value is a bool.
        If domain_type is unknown (None), value is None.
    """
    hyperparameter_name: str
    domain_type: Union[HyperparameterDomainType, None] = None
    value: Union[float, str, bool, None] = None