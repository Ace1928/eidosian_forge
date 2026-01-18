from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def DECLARE(name: str, memory_type: str='BIT', memory_size: int=1, shared_region: Optional[str]=None, offsets: Optional[Iterable[Tuple[int, str]]]=None) -> Declare:
    return Declare(name=name, memory_type=memory_type, memory_size=memory_size, shared_region=shared_region, offsets=offsets)