import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
class _DataPipeSerializationWrapper:

    def __init__(self, datapipe):
        self._datapipe = datapipe

    def __getstate__(self):
        use_dill = False
        try:
            value = pickle.dumps(self._datapipe)
        except Exception:
            if HAS_DILL:
                value = dill.dumps(self._datapipe)
                use_dill = True
            else:
                raise
        return (value, use_dill)

    def __setstate__(self, state):
        value, use_dill = state
        if use_dill:
            self._datapipe = dill.loads(value)
        else:
            self._datapipe = pickle.loads(value)

    def __len__(self):
        try:
            return len(self._datapipe)
        except Exception as e:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length") from e