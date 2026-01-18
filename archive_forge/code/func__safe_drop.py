from __future__ import annotations
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Sequence
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.exceptions import EntityExistsError, NoEngineError
def _safe_drop(self) -> None:
    """
        Run an existence check before attempting to drop the entity from the cluster.
        """
    if self._exists():
        self._drop()
    elif self.check_if_exists or self.check_if_any_exist:
        raise EntityExistsError(f'There is no {self.__class__.__name__} with the name {self.name} to remove. If you want to proceed anyway, set the `check_if_exists` parameter to False. This will simply skip over the removal of this entity that does not exist in the cluster.')