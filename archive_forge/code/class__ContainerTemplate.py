import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
class _ContainerTemplate(ABC):
    """Abstract class for container ``DataPipes``. The followings are three required methods."""

    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int):
        ...

    @abstractmethod
    def is_every_instance_exhausted(self) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        """Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""