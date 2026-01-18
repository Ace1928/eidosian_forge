import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterator, List, Optional, Sized, TypeVar
import torch.utils.data.datapipes.iter.sharding
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
def _dive(self, element, unbatch_level):
    if unbatch_level < -1:
        raise ValueError('unbatch_level must be -1 or >= 0')
    if unbatch_level == -1:
        if isinstance(element, (list, DataChunk)):
            for item in element:
                yield from self._dive(item, unbatch_level=-1)
        else:
            yield element
    elif unbatch_level == 0:
        yield element
    elif isinstance(element, (list, DataChunk)):
        for item in element:
            yield from self._dive(item, unbatch_level=unbatch_level - 1)
    else:
        raise IndexError(f'unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe')