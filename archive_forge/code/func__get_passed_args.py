from __future__ import annotations
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Sequence
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.exceptions import EntityExistsError, NoEngineError
def _get_passed_args(self) -> dict[str, Any]:
    """
        Helper to grab all the arguments to __init__ that aren't in the superclass and have a non-None value. Useful
        for subclasses.
        :return: A dict mapping the names of init arguments to their values.
        """
    return {k: v for k, v in vars(self).items() if k not in signature(self.__class__.__bases__[0].__init__).parameters and v is not None and (k != '_grant_name')}