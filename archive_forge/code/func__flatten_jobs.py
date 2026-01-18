import abc
from typing import Any, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
import duet
import numpy as np
from cirq import study, value
def _flatten_jobs(tree: Optional[CIRCUIT_SAMPLE_JOB_TREE]) -> Iterator[CircuitSampleJob]:
    if isinstance(tree, CircuitSampleJob):
        yield tree
    elif tree is not None:
        for item in tree:
            yield from _flatten_jobs(item)