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
def _upload_worker_data(self):
    """
        Uploads worker data acceptance and completion rates to the parlai server.
        """
    worker_data = self.worker_manager.get_worker_data_package()
    data = {'worker_data': worker_data}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    try:
        requests.post(PARLAI_MTURK_UPLOAD_URL, json=data, headers=headers)
    except Exception:
        shared_utils.print_and_log(logging.WARNING, 'Unable to log worker statistics to parl.ai', should_print=True)