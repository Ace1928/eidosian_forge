from kombu import exceptions as kombu_exc
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def _collect_requeue_votes(self, data, message):
    requeue_votes = 0
    for i, cb in enumerate(self._requeue_filters):
        try:
            if cb(data, message):
                requeue_votes += 1
        except Exception:
            LOG.exception("Failed calling requeue filter %s '%s' to determine if message %r should be requeued.", i + 1, cb, message.delivery_tag)
    return requeue_votes