from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
def _get_judge(self, trial: Trial) -> _PerPartition:
    key = to_uuid(trial.keys)
    with self._lock:
        if key not in self._data:
            self._data[key] = _PerPartition(self, trial.keys)
        return self._data[key]