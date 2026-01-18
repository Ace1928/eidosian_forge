from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class HyperparameterSessionRun:
    """A single run in a HyperparameterSessionGroup.

    Attributes:
      experiment_id: The id of the experiment to which the run belongs.
      run: The name of the run.
    """
    experiment_id: str
    run: str