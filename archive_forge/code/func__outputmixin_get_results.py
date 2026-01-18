from abc import ABC, abstractmethod
from typing import Mapping, Any
@abstractmethod
def _outputmixin_get_results(self) -> Mapping[str, Any]:
    """Return Mapping of names to result value.

        This may be called many times and should hence not be
        expensive (except possibly the first time)."""