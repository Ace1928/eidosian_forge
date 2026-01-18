import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def _set_cache_by_message(self, task_id, message):
    payload = self._cache[task_id] = self.meta_from_decoded(message.payload)
    return payload