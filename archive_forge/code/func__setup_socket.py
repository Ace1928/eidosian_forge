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
def _setup_socket(self, timeout_seconds=None):
    """
        Set up a socket_manager with defined callbacks.
        """
    assert self.task_state >= self.STATE_INIT_RUN, 'socket cannot be set up until run is started'
    socket_server_url = self.server_url
    if self.opt['local']:
        socket_server_url = 'https://localhost'
    self.socket_manager = SocketManager(socket_server_url, self.port, self._on_alive, self._on_new_message, self._on_socket_dead, self.task_group_id, socket_dead_timeout=timeout_seconds, server_death_callback=self.shutdown)