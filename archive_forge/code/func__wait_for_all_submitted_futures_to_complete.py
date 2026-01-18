import copy
import logging
from s3transfer.utils import get_callbacks
def _wait_for_all_submitted_futures_to_complete(self):
    submitted_futures = self._transfer_coordinator.associated_futures
    while submitted_futures:
        self._wait_until_all_complete(submitted_futures)
        possibly_more_submitted_futures = self._transfer_coordinator.associated_futures
        if submitted_futures == possibly_more_submitted_futures:
            break
        submitted_futures = possibly_more_submitted_futures