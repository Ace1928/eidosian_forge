import functools
import threading
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import dispatcher
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import proxy
from taskflow.engines.worker_based import types as wt
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.task import EVENT_UPDATE_PROGRESS  # noqa
from taskflow.utils import kombu_utils as ku
from taskflow.utils import misc
from taskflow.utils import threading_utils as tu
@staticmethod
def _handle_expired_request(request):
    """Handle a expired request.

        When a request has expired it is removed from the ongoing requests
        dictionary and a ``RequestTimeout`` exception is set as a
        request result.
        """
    if request.transition_and_log_error(pr.FAILURE, logger=LOG):
        try:
            request_age = timeutils.now() - request.created_on
            raise exc.RequestTimeout("Request '%s' has expired after waiting for %0.2f seconds for it to transition out of (%s) states" % (request, request_age, ', '.join(pr.WAITING_STATES)))
        except exc.RequestTimeout:
            with misc.capture_failure() as failure:
                LOG.debug(failure.exception_str)
                request.set_result(failure)
        return True
    return False