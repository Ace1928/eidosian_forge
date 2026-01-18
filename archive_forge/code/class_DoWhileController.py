from __future__ import annotations
import logging
from collections.abc import Callable, Iterable, Generator
from typing import Any
from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError
class DoWhileController(BaseController):
    """Run the given tasks in a loop until the ``do_while`` condition on the property set becomes
    ``False``.

    The given tasks will always run at least once, and on iteration of the loop, all the
    tasks will be run (with the exception of a failure state being set)."""

    def __init__(self, tasks: Task | Iterable[Task]=(), do_while: Callable[[PropertySet], bool]=None, *, options: dict[str, Any] | None=None):
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.do_while = do_while

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        max_iteration = self._options.get('max_iteration', 1000)
        for _ in range(max_iteration):
            for task in self.tasks:
                state = (yield task)
            if not self.do_while(state.property_set):
                return
            state.workflow_status.completed_passes.difference_update(self.tasks)
        raise PassManagerError('Maximum iteration reached. max_iteration=%i' % max_iteration)