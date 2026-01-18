import numbers
from collections import namedtuple
from collections.abc import Mapping
from datetime import timedelta
from weakref import WeakValueDictionary
from kombu import Connection, Consumer, Exchange, Producer, Queue, pools
from kombu.common import Broadcast
from kombu.utils.functional import maybe_list
from kombu.utils.objects import cached_property
from celery import signals
from celery.utils.nodenames import anon_nodename
from celery.utils.saferepr import saferepr
from celery.utils.text import indent as textindent
from celery.utils.time import maybe_make_aware
from . import routes as _routes
def Queues(self, queues, create_missing=None, autoexchange=None, max_priority=None):
    conf = self.app.conf
    default_routing_key = conf.task_default_routing_key
    if create_missing is None:
        create_missing = conf.task_create_missing_queues
    if max_priority is None:
        max_priority = conf.task_queue_max_priority
    if not queues and conf.task_default_queue:
        queues = (Queue(conf.task_default_queue, exchange=self.default_exchange, routing_key=default_routing_key),)
    autoexchange = self.autoexchange if autoexchange is None else autoexchange
    return self.queues_cls(queues, self.default_exchange, create_missing, autoexchange, max_priority, default_routing_key)