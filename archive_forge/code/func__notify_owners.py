from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
def _notify_owners(self, old: Any, hint: DocumentPatchedEvent | None=None) -> None:
    for owner, descriptor in self._owners:
        descriptor._notify_mutated(owner, old, hint=hint)