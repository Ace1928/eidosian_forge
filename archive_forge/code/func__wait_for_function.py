import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def _wait_for_function(self, function_descriptor, job_id: str, timeout=10):
    """Wait until the function to be executed is present on this worker.
        This method will simply loop until the import thread has imported the
        relevant function. If we spend too long in this loop, that may indicate
        a problem somewhere and we will push an error message to the user.
        If this worker is an actor, then this will wait until the actor has
        been defined.
        Args:
            function_descriptor : The FunctionDescriptor of the function that
                we want to execute.
            job_id: The ID of the job to push the error message to
                if this times out.
        """
    start_time = time.time()
    warning_sent = False
    while True:
        with self.lock:
            if self._worker.actor_id.is_nil():
                if function_descriptor.function_id in self._function_execution_info:
                    break
                else:
                    key = make_function_table_key(b'RemoteFunction', job_id, function_descriptor.function_id.binary())
                    if self.fetch_and_register_remote_function(key) is True:
                        break
            else:
                assert not self._worker.actor_id.is_nil()
                assert self._worker.actor_id in self._worker.actors
                break
        if time.time() - start_time > timeout:
            warning_message = f'This worker was asked to execute a function that has not been registered ({function_descriptor}, node={self._worker.node_ip_address}, worker_id={self._worker.worker_id.hex()}, pid={os.getpid()}). You may have to restart Ray.'
            if not warning_sent:
                logger.error(warning_message)
                ray._private.utils.push_error_to_driver(self._worker, ray_constants.WAIT_FOR_FUNCTION_PUSH_ERROR, warning_message, job_id=job_id)
            warning_sent = True
        self._worker.import_thread._do_importing()
        time.sleep(0.001)