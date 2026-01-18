from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class Hyperparameter:
    """Metadata about a hyperparameter.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter that
        should be unique in any result set of Hyperparameter objects.
      hyperparameter_display_name: A displayable name for the hyperparameter.
        Unlike hyperparameter_name, there is no uniqueness constraint.
      domain_type: A HyperparameterDomainType describing how we represent the
        set of known values in the `domain` attribute.
      domain: A representation of the set of known values for the
        hyperparameter.

        If domain_type is INTERVAL, a Tuple[float, float] describing the
          range of numeric values.
        If domain_type is DISCRETE_FLOAT, a Collection[float] describing the
          finite set of numeric values.
        If domain_type is DISCRETE_STRING, a Collection[string] describing the
          finite set of string values.
        If domain_type is DISCRETE_BOOL, a Collection[bool] describing the
          finite set of bool values.

      differs: Describes whether there are two or more known values for the
        hyperparameter for the set of experiments specified in the
        list_hyperparameters() request. Hyperparameters for which this is
        true are made more prominent or easier to discover in the UI.
    """
    hyperparameter_name: str
    hyperparameter_display_name: str
    domain_type: Union[HyperparameterDomainType, None] = None
    domain: Union[Tuple[float, float], Collection[float], Collection[str], Collection[bool], None] = None
    differs: bool = False