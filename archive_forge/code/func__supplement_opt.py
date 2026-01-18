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
def _supplement_opt(self):
    """
        Add additional args to opt.

        Useful to add relevant options after args are parsed.
        """
    self.opt.update({'task': os.path.basename(os.path.dirname(os.path.abspath(__file__))), 'task_description': {'num_subtasks': self.opt['subtasks_per_hit'], 'question': self.opt['question']}, 'frontend_version': 1})
    self.opt.update(self.opt['task_config'])