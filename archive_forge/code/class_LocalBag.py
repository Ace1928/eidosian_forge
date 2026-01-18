from abc import abstractmethod
from typing import Any, List, Optional
from ..dataset import Dataset, DatasetDisplay, get_dataset_display
class LocalBag(Bag):

    @property
    def is_local(self) -> bool:
        return True

    @property
    def num_partitions(self) -> int:
        return 1