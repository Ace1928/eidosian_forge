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
def _extract_info(task):
    fields = {field: str(getattr(task, field, None)) for field in taskinfoitems if getattr(task, field, None) is not None}
    if fields:
        info = ['='.join(f) for f in fields.items()]
        return '{} [{}]'.format(task.name, ' '.join(info))
    return task.name