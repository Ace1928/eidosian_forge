import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset

        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        