import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
@staticmethod
def _match_worker(task, available_workers):
    """Select a worker (from geq 1 workers) that can best perform the task.

        NOTE(harlowja): this method will be activated when there exists
        one one greater than one potential workers that can perform a task,
        the arguments provided will be the potential workers located and the
        task that is being requested to perform and the result should be one
        of those workers using whatever best-fit algorithm is possible (or
        random at the least).
        """
    if len(available_workers) == 1:
        return available_workers[0]
    else:
        return random.choice(available_workers)