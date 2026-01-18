import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _expire_agent_pool(self):
    """
        Expire all workers in the worker pool.
        """
    for agent in self.agent_pool.copy():
        self.force_expire_hit(agent.worker_id, agent.assignment_id)
        with self.agent_pool_change_condition:
            self._remove_from_agent_pool(agent)