from abc import abstractmethod
from typing import Any, List, Optional
from ..dataset import Dataset, DatasetDisplay, get_dataset_display
class LocalBoundedBag(LocalBag):

    @property
    def is_bounded(self) -> bool:
        return True

    def as_local_bounded(self) -> 'LocalBoundedBag':
        return self