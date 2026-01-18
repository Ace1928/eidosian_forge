from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def _top_up_task_data(self, worker_id: str, task_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
        Top up worker task data.

        This function is called if ``self.task_queue`` is exhausted but
        task_data for the worker is less than the `tasks_per_hit`.

        Make sure that all added tasks have not been seen by the worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
    worker_data = self._get_worker_data(worker_id)
    tasks_still_needed = self.opt['subtasks_per_hit'] - len(task_data)
    tasks_remaining = [t_id for t_id in range(len(self.desired_tasks)) if t_id not in worker_data['tasks_completed']]
    additional_tasks = [t for t in tasks_remaining if all((d_id not in worker_data['conversations_seen'] for d_id in self._get_dialogue_ids(self.desired_tasks[t])))]
    if tasks_still_needed < len(additional_tasks):
        additional_tasks = random.sample(additional_tasks, tasks_still_needed)
    worker_data['tasks_completed'].extend(additional_tasks)
    for t in additional_tasks:
        worker_data['conversations_seen'].extend(self._get_dialogue_ids(self.desired_tasks[t]))
        task_data.append(self.desired_tasks[t])
    return task_data