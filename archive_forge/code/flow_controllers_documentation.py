from __future__ import annotations
import logging
from collections.abc import Callable, Iterable, Generator
from typing import Any
from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError
Alias of tasks for backward compatibility.