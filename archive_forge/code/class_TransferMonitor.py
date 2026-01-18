import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
class TransferMonitor(object):

    def __init__(self):
        """Monitors transfers for cross-proccess communication

        Notifications can be sent to the monitor and information can be
        retrieved from the monitor for a particular transfer. This abstraction
        is ran in a ``multiprocessing.managers.BaseManager`` in order to be
        shared across processes.
        """
        self._transfer_states = {}
        self._id_count = 0
        self._init_lock = threading.Lock()

    def notify_new_transfer(self):
        with self._init_lock:
            transfer_id = self._id_count
            self._transfer_states[transfer_id] = TransferState()
            self._id_count += 1
            return transfer_id

    def is_done(self, transfer_id):
        """Determine a particular transfer is complete

        :param transfer_id: Unique identifier for the transfer
        :return: True, if done. False, otherwise.
        """
        return self._transfer_states[transfer_id].done

    def notify_done(self, transfer_id):
        """Notify a particular transfer is complete

        :param transfer_id: Unique identifier for the transfer
        """
        self._transfer_states[transfer_id].set_done()

    def poll_for_result(self, transfer_id):
        """Poll for the result of a transfer

        :param transfer_id: Unique identifier for the transfer
        :return: If the transfer succeeded, it will return the result. If the
            transfer failed, it will raise the exception associated to the
            failure.
        """
        self._transfer_states[transfer_id].wait_till_done()
        exception = self._transfer_states[transfer_id].exception
        if exception:
            raise exception
        return None

    def notify_exception(self, transfer_id, exception):
        """Notify an exception was encountered for a transfer

        :param transfer_id: Unique identifier for the transfer
        :param exception: The exception encountered for that transfer
        """
        self._transfer_states[transfer_id].exception = exception

    def notify_cancel_all_in_progress(self):
        for transfer_state in self._transfer_states.values():
            if not transfer_state.done:
                transfer_state.exception = CancelledError()

    def get_exception(self, transfer_id):
        """Retrieve the exception encountered for the transfer

        :param transfer_id: Unique identifier for the transfer
        :return: The exception encountered for that transfer. Otherwise
            if there were no exceptions, returns None.
        """
        return self._transfer_states[transfer_id].exception

    def notify_expected_jobs_to_complete(self, transfer_id, num_jobs):
        """Notify the amount of jobs expected for a transfer

        :param transfer_id: Unique identifier for the transfer
        :param num_jobs: The number of jobs to complete the transfer
        """
        self._transfer_states[transfer_id].jobs_to_complete = num_jobs

    def notify_job_complete(self, transfer_id):
        """Notify that a single job is completed for a transfer

        :param transfer_id: Unique identifier for the transfer
        :return: The number of jobs remaining to complete the transfer
        """
        return self._transfer_states[transfer_id].decrement_jobs_to_complete()