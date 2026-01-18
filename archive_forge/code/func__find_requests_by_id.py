import io
import tempfile
from collections import UserDict, defaultdict, namedtuple
from billiard.common import TERM_SIGNAME
from kombu.utils.encoding import safe_repr
from celery.exceptions import WorkerShutdown
from celery.platforms import signals as _signals
from celery.utils.functional import maybe_list
from celery.utils.log import get_logger
from celery.utils.serialization import jsonify, strtobool
from celery.utils.time import rate
from . import state as worker_state
from .request import Request
def _find_requests_by_id(ids, get_request=worker_state.requests.__getitem__):
    for task_id in ids:
        try:
            yield get_request(task_id)
        except KeyError:
            pass