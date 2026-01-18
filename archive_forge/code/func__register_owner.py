from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
def _register_owner(self, owner: HasProps, descriptor: PropertyDescriptor[Any]) -> None:
    self._owners.add((owner, descriptor))