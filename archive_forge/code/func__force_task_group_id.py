import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def _force_task_group_id(self, task_group_id):
    """
        Throw an error if a task group id is neither provided nor stored.
        """
    if task_group_id is None:
        task_group_id = self.task_group_id
    assert task_group_id is not None, 'Default task_group_id not set'
    return task_group_id