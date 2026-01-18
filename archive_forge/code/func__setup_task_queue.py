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
def _setup_task_queue(self):
    """
        Fill task queue with conversation pairs.
        """
    for _i in range(self.opt['annotations_per_pair']):
        all_task_keys = list(range(len(self.desired_tasks)))
        random.shuffle(all_task_keys)
        for p_id in all_task_keys:
            self.task_queue.put(self.desired_tasks[p_id])