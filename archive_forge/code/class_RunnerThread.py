import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
class RunnerThread(threading.Thread):
    """Supervisor thread that runs your script."""

    def __init__(self, *args, error_queue, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self._error_queue = error_queue
        self._ret = None

    def _propagate_exception(self, e: BaseException):
        try:
            self._error_queue.put(e, block=True, timeout=_ERROR_REPORT_TIMEOUT)
        except queue.Full:
            logger.critical('Runner Thread was unable to report error to main function runner thread. This means a previous error was not processed. This should never happen.')

    def run(self):
        try:
            self._ret = self._target(*self._args, **self._kwargs)
        except StopIteration:
            logger.debug('Thread runner raised StopIteration. Interpreting it as a signal to terminate the thread without error.')
        except SystemExit as e:
            if e.code == 0:
                logger.debug('Thread runner raised SystemExit with error code 0. Interpreting it as a signal to terminate the thread without error.')
            else:
                self._propagate_exception(e)
        except BaseException as e:
            self._propagate_exception(e)

    def join(self, timeout=None):
        super(RunnerThread, self).join(timeout)
        return self._ret