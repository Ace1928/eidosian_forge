import functools
import threading
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import dispatcher
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import proxy
from taskflow.engines.worker_based import types as wt
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.task import EVENT_UPDATE_PROGRESS  # noqa
from taskflow.utils import kombu_utils as ku
from taskflow.utils import misc
from taskflow.utils import threading_utils as tu
def _on_wait(self):
    """This function is called cyclically between draining events."""
    self._finder.maybe_publish()
    self._finder.clean()
    self._clean()