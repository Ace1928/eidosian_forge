from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def _update_max_size(self, max_size: int, after_cleanup_size: Optional[int]=None) -> None:
    self._max_size = max_size
    if after_cleanup_size is None:
        self._after_cleanup_size = self._max_size * 8 // 10
    else:
        self._after_cleanup_size = min(after_cleanup_size, self._max_size)