from abc import abstractmethod
from typing import Any, List, Optional
from ..dataset import Dataset, DatasetDisplay, get_dataset_display
class BagDisplay(DatasetDisplay):
    """:class:`~.Bag` plain display class"""

    @property
    def bg(self) -> Bag:
        """The target :class:`~.Bag`"""
        return self._ds

    def show(self, n: int=10, with_count: bool=False, title: Optional[str]=None) -> None:
        head_rows = self.bg.head(n).as_array()
        if len(head_rows) < n:
            count = len(head_rows)
        else:
            count = self.bg.count() if with_count else -1
        with DatasetDisplay._SHOW_LOCK:
            if title is not None and title != '':
                print(title)
            print(type(self.bg).__name__)
            print(head_rows)
            if count >= 0:
                print(f'Total count: {count}')
                print('')
            if self.bg.has_metadata:
                print('Metadata:')
                try:
                    print(self.bg.metadata.to_json(indent=True))
                except Exception:
                    print(self.bg.metadata)
                print('')