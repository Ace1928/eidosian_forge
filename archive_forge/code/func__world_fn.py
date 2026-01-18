import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
def _world_fn():
    log_utils.print_and_log(logging.INFO, 'Starting task {}...'.format(task_name))
    return self._run_world(task, world_name, agents)