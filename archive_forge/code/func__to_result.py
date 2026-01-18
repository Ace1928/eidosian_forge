import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def _to_result(self, task_id, state, result, traceback, request):
    return {'task_id': task_id, 'status': state, 'result': self.encode_result(result, state), 'traceback': traceback, 'children': self.current_task_children(request)}