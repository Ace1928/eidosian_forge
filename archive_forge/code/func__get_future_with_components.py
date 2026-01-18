import copy
import logging
import threading
from botocore.compat import six
from s3transfer.constants import KB, MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.utils import get_callbacks
from s3transfer.utils import signal_transferring
from s3transfer.utils import signal_not_transferring
from s3transfer.utils import CallArgs
from s3transfer.utils import OSUtils
from s3transfer.utils import TaskSemaphore
from s3transfer.utils import SlidingWindowSemaphore
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import FatalError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.futures import BoundedExecutor
from s3transfer.futures import TransferFuture
from s3transfer.futures import TransferMeta
from s3transfer.futures import TransferCoordinator
from s3transfer.download import DownloadSubmissionTask
from s3transfer.upload import UploadSubmissionTask
from s3transfer.copies import CopySubmissionTask
from s3transfer.delete import DeleteSubmissionTask
from s3transfer.bandwidth import LeakyBucket
from s3transfer.bandwidth import BandwidthLimiter
def _get_future_with_components(self, call_args):
    transfer_id = self._id_counter
    transfer_coordinator = TransferCoordinator(transfer_id=transfer_id)
    self._coordinator_controller.add_transfer_coordinator(transfer_coordinator)
    transfer_coordinator.add_done_callback(self._coordinator_controller.remove_transfer_coordinator, transfer_coordinator)
    components = {'meta': TransferMeta(call_args, transfer_id=transfer_id), 'coordinator': transfer_coordinator}
    transfer_future = TransferFuture(**components)
    return (transfer_future, components)