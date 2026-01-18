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
class AMQP:
    """App AMQP API: app.amqp."""
    Connection = Connection
    Consumer = Consumer
    Producer = Producer
    BrokerConnection = Connection
    queues_cls = Queues
    _rtable = None
    _producer_pool = None
    autoexchange = None
    argsrepr_maxsize = 1024
    kwargsrepr_maxsize = 1024

    def __init__(self, app):
        self.app = app
        self.task_protocols = {1: self.as_task_v1, 2: self.as_task_v2}
        self.app._conf.bind_to(self._handle_conf_update)

    @cached_property
    def create_task_message(self):
        return self.task_protocols[self.app.conf.task_protocol]

    @cached_property
    def send_task_message(self):
        return self._create_task_sender()

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

    def Router(self, queues=None, create_missing=None):
        """Return the current task router."""
        return _routes.Router(self.routes, queues or self.queues, self.app.either('task_create_missing_queues', create_missing), app=self.app)

    def flush_routes(self):
        self._rtable = _routes.prepare(self.app.conf.task_routes)

    def TaskConsumer(self, channel, queues=None, accept=None, **kw):
        if accept is None:
            accept = self.app.conf.accept_content
        return self.Consumer(channel, accept=accept, queues=queues or list(self.queues.consume_from.values()), **kw)

    def as_task_v2(self, task_id, name, args=None, kwargs=None, countdown=None, eta=None, group_id=None, group_index=None, expires=None, retries=0, chord=None, callbacks=None, errbacks=None, reply_to=None, time_limit=None, soft_time_limit=None, create_sent_event=False, root_id=None, parent_id=None, shadow=None, chain=None, now=None, timezone=None, origin=None, ignore_result=False, argsrepr=None, kwargsrepr=None, stamped_headers=None, replaced_task_nesting=0, **options):
        args = args or ()
        kwargs = kwargs or {}
        if not isinstance(args, (list, tuple)):
            raise TypeError('task args must be a list or tuple')
        if not isinstance(kwargs, Mapping):
            raise TypeError('task keyword arguments must be a mapping')
        if countdown:
            self._verify_seconds(countdown, 'countdown')
            now = now or self.app.now()
            timezone = timezone or self.app.timezone
            eta = maybe_make_aware(now + timedelta(seconds=countdown), tz=timezone)
        if isinstance(expires, numbers.Real):
            self._verify_seconds(expires, 'expires')
            now = now or self.app.now()
            timezone = timezone or self.app.timezone
            expires = maybe_make_aware(now + timedelta(seconds=expires), tz=timezone)
        if not isinstance(eta, str):
            eta = eta and eta.isoformat()
        if not isinstance(expires, str):
            expires = expires and expires.isoformat()
        if argsrepr is None:
            argsrepr = saferepr(args, self.argsrepr_maxsize)
        if kwargsrepr is None:
            kwargsrepr = saferepr(kwargs, self.kwargsrepr_maxsize)
        if not root_id:
            root_id = task_id
        stamps = {header: options[header] for header in stamped_headers or []}
        headers = {'lang': 'py', 'task': name, 'id': task_id, 'shadow': shadow, 'eta': eta, 'expires': expires, 'group': group_id, 'group_index': group_index, 'retries': retries, 'timelimit': [time_limit, soft_time_limit], 'root_id': root_id, 'parent_id': parent_id, 'argsrepr': argsrepr, 'kwargsrepr': kwargsrepr, 'origin': origin or anon_nodename(), 'ignore_result': ignore_result, 'replaced_task_nesting': replaced_task_nesting, 'stamped_headers': stamped_headers, 'stamps': stamps}
        return task_message(headers=headers, properties={'correlation_id': task_id, 'reply_to': reply_to or ''}, body=(args, kwargs, {'callbacks': callbacks, 'errbacks': errbacks, 'chain': chain, 'chord': chord}), sent_event={'uuid': task_id, 'root_id': root_id, 'parent_id': parent_id, 'name': name, 'args': argsrepr, 'kwargs': kwargsrepr, 'retries': retries, 'eta': eta, 'expires': expires} if create_sent_event else None)

    def as_task_v1(self, task_id, name, args=None, kwargs=None, countdown=None, eta=None, group_id=None, group_index=None, expires=None, retries=0, chord=None, callbacks=None, errbacks=None, reply_to=None, time_limit=None, soft_time_limit=None, create_sent_event=False, root_id=None, parent_id=None, shadow=None, now=None, timezone=None, **compat_kwargs):
        args = args or ()
        kwargs = kwargs or {}
        utc = self.utc
        if not isinstance(args, (list, tuple)):
            raise TypeError('task args must be a list or tuple')
        if not isinstance(kwargs, Mapping):
            raise TypeError('task keyword arguments must be a mapping')
        if countdown:
            self._verify_seconds(countdown, 'countdown')
            now = now or self.app.now()
            eta = now + timedelta(seconds=countdown)
        if isinstance(expires, numbers.Real):
            self._verify_seconds(expires, 'expires')
            now = now or self.app.now()
            expires = now + timedelta(seconds=expires)
        eta = eta and eta.isoformat()
        expires = expires and expires.isoformat()
        return task_message(headers={}, properties={'correlation_id': task_id, 'reply_to': reply_to or ''}, body={'task': name, 'id': task_id, 'args': args, 'kwargs': kwargs, 'group': group_id, 'group_index': group_index, 'retries': retries, 'eta': eta, 'expires': expires, 'utc': utc, 'callbacks': callbacks, 'errbacks': errbacks, 'timelimit': (time_limit, soft_time_limit), 'taskset': group_id, 'chord': chord}, sent_event={'uuid': task_id, 'name': name, 'args': saferepr(args), 'kwargs': saferepr(kwargs), 'retries': retries, 'eta': eta, 'expires': expires} if create_sent_event else None)

    def _verify_seconds(self, s, what):
        if s < INT_MIN:
            raise ValueError(f'{what} is out of range: {s!r}')
        return s

    def _create_task_sender(self):
        default_retry = self.app.conf.task_publish_retry
        default_policy = self.app.conf.task_publish_retry_policy
        default_delivery_mode = self.app.conf.task_default_delivery_mode
        default_queue = self.default_queue
        queues = self.queues
        send_before_publish = signals.before_task_publish.send
        before_receivers = signals.before_task_publish.receivers
        send_after_publish = signals.after_task_publish.send
        after_receivers = signals.after_task_publish.receivers
        send_task_sent = signals.task_sent.send
        sent_receivers = signals.task_sent.receivers
        default_evd = self._event_dispatcher
        default_exchange = self.default_exchange
        default_rkey = self.app.conf.task_default_routing_key
        default_serializer = self.app.conf.task_serializer
        default_compressor = self.app.conf.task_compression

        def send_task_message(producer, name, message, exchange=None, routing_key=None, queue=None, event_dispatcher=None, retry=None, retry_policy=None, serializer=None, delivery_mode=None, compression=None, declare=None, headers=None, exchange_type=None, **kwargs):
            retry = default_retry if retry is None else retry
            headers2, properties, body, sent_event = message
            if headers:
                headers2.update(headers)
            if kwargs:
                properties.update(kwargs)
            qname = queue
            if queue is None and exchange is None:
                queue = default_queue
            if queue is not None:
                if isinstance(queue, str):
                    qname, queue = (queue, queues[queue])
                else:
                    qname = queue.name
            if delivery_mode is None:
                try:
                    delivery_mode = queue.exchange.delivery_mode
                except AttributeError:
                    pass
                delivery_mode = delivery_mode or default_delivery_mode
            if exchange_type is None:
                try:
                    exchange_type = queue.exchange.type
                except AttributeError:
                    exchange_type = 'direct'
            if (not exchange or not routing_key) and exchange_type == 'direct':
                exchange, routing_key = ('', qname)
            elif exchange is None:
                exchange = queue.exchange.name or default_exchange
                routing_key = routing_key or queue.routing_key or default_rkey
            if declare is None and queue and (not isinstance(queue, Broadcast)):
                declare = [queue]
            retry = default_retry if retry is None else retry
            _rp = dict(default_policy, **retry_policy) if retry_policy else default_policy
            if before_receivers:
                send_before_publish(sender=name, body=body, exchange=exchange, routing_key=routing_key, declare=declare, headers=headers2, properties=properties, retry_policy=retry_policy)
            ret = producer.publish(body, exchange=exchange, routing_key=routing_key, serializer=serializer or default_serializer, compression=compression or default_compressor, retry=retry, retry_policy=_rp, delivery_mode=delivery_mode, declare=declare, headers=headers2, **properties)
            if after_receivers:
                send_after_publish(sender=name, body=body, headers=headers2, exchange=exchange, routing_key=routing_key)
            if sent_receivers:
                if isinstance(body, tuple):
                    send_task_sent(sender=name, task_id=headers2['id'], task=name, args=body[0], kwargs=body[1], eta=headers2['eta'], taskset=headers2['group'])
                else:
                    send_task_sent(sender=name, task_id=body['id'], task=name, args=body['args'], kwargs=body['kwargs'], eta=body['eta'], taskset=body['taskset'])
            if sent_event:
                evd = event_dispatcher or default_evd
                exname = exchange
                if isinstance(exname, Exchange):
                    exname = exname.name
                sent_event.update({'queue': qname, 'exchange': exname, 'routing_key': routing_key})
                evd.publish('task-sent', sent_event, producer, retry=retry, retry_policy=retry_policy)
            return ret
        return send_task_message

    @cached_property
    def default_queue(self):
        return self.queues[self.app.conf.task_default_queue]

    @cached_property
    def queues(self):
        """Queue name⇒ declaration mapping."""
        return self.Queues(self.app.conf.task_queues)

    @queues.setter
    def queues(self, queues):
        return self.Queues(queues)

    @property
    def routes(self):
        if self._rtable is None:
            self.flush_routes()
        return self._rtable

    @cached_property
    def router(self):
        return self.Router()

    @router.setter
    def router(self, value):
        return value

    @property
    def producer_pool(self):
        if self._producer_pool is None:
            self._producer_pool = pools.producers[self.app.connection_for_write()]
            self._producer_pool.limit = self.app.pool.limit
        return self._producer_pool
    publisher_pool = producer_pool

    @cached_property
    def default_exchange(self):
        return Exchange(self.app.conf.task_default_exchange, self.app.conf.task_default_exchange_type)

    @cached_property
    def utc(self):
        return self.app.conf.enable_utc

    @cached_property
    def _event_dispatcher(self):
        return self.app.events.Dispatcher(enabled=False)

    def _handle_conf_update(self, *args, **kwargs):
        if 'task_routes' in kwargs or 'task_routes' in args:
            self.flush_routes()
            self.router = self.Router()
        return