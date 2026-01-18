from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class HyperparameterSessionGroup:
    """A group of runs logically executed together with the same hparam values.

    The group of runs may have, for example, combined to generate and test a
    single model or other artifacts.

    We assume these groups of runs were executed with the same set of
    hyperparameter values. However, having the same set of hyperparameter values
    is not sufficient to be considered part of the same group -- different
    groups can exist with the same hyperparameter values.

    Attributes:
      root: A descriptor of the common ancestor of all sessions in this
        group.

        In the case where the group contains all runs in the experiment, this
        would just be a HyperparameterSessionRun with the experiment_id property
        set to the experiment's id but run property set to empty.

        In the case where the group contains a subset of runs in the experiment,
        this would be a HyperparameterSessionRun with the experiment_id property
        set and the run property set to the largest common prefix for runs.

        The root might correspond to a session within the group but it is not
        necessary.
      sessions: A sequence of all sessions in this group.
      hyperparameter_values: A collection of all hyperparameter values in this
        group.
    """
    root: HyperparameterSessionRun
    sessions: Sequence[HyperparameterSessionRun]
    hyperparameter_values: Collection[HyperparameterValue]