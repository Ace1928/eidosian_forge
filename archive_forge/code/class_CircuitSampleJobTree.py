import abc
from typing import Any, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
import duet
import numpy as np
from cirq import study, value
class CircuitSampleJobTree(Protocol):

    def __iter__(self) -> Iterator[Union[CircuitSampleJob, 'CircuitSampleJobTree']]:
        pass