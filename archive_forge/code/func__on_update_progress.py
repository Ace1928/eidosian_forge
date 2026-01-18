import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def _on_update_progress(self, task, event_type, details):
    """Should be called when task updates its progress."""
    try:
        progress = details.pop('progress')
    except KeyError:
        pass
    else:
        try:
            self._storage.set_task_progress(task.name, progress, details=details)
        except Exception:
            LOG.exception('Failed setting task progress for %s to %0.3f', task, progress)