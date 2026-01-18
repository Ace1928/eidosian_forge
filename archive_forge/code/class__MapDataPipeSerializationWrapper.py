import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):

    def __getitem__(self, idx):
        return self._datapipe[idx]