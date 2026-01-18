from __future__ import annotations
import logging
from collections.abc import Callable, Iterable, Generator
from typing import Any
from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError
class FlowControllerLinear(BaseController):
    """A standard flow controller that runs tasks one after the other."""

    def __init__(self, tasks: Task | Iterable[Task]=(), *, options: dict[str, Any] | None=None):
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        for task in self.tasks:
            state = (yield task)