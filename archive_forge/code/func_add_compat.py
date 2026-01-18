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
def add_compat(self, name, **options):
    options.setdefault('routing_key', options.get('binding_key'))
    if options['routing_key'] is None:
        options['routing_key'] = name
    return self._add(Queue.from_dict(name, **options))