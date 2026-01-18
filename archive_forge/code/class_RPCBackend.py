import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
class RPCBackend(base.Backend, AsyncBackendMixin):
    """Base class for the RPC result backend."""
    Exchange = kombu.Exchange
    Producer = kombu.Producer
    ResultConsumer = ResultConsumer
    BacklogLimitExceeded = BacklogLimitExceeded
    persistent = False
    supports_autoexpire = True
    supports_native_join = True
    retry_policy = {'max_retries': 20, 'interval_start': 0, 'interval_step': 1, 'interval_max': 1}

    class Consumer(kombu.Consumer):
        """Consumer that requires manual declaration of queues."""
        auto_declare = False

    class Queue(kombu.Queue):
        """Queue that never caches declaration."""
        can_cache_declaration = False

    def __init__(self, app, connection=None, exchange=None, exchange_type=None, persistent=None, serializer=None, auto_delete=True, **kwargs):
        super().__init__(app, **kwargs)
        conf = self.app.conf
        self._connection = connection
        self._out_of_band = {}
        self.persistent = self.prepare_persistent(persistent)
        self.delivery_mode = 2 if self.persistent else 1
        exchange = exchange or conf.result_exchange
        exchange_type = exchange_type or conf.result_exchange_type
        self.exchange = self._create_exchange(exchange, exchange_type, self.delivery_mode)
        self.serializer = serializer or conf.result_serializer
        self.auto_delete = auto_delete
        self.result_consumer = self.ResultConsumer(self, self.app, self.accept, self._pending_results, self._pending_messages)
        if register_after_fork is not None:
            register_after_fork(self, _on_after_fork_cleanup_backend)

    def _after_fork(self):
        self._pending_results.clear()
        self.result_consumer._after_fork()

    def _create_exchange(self, name, type='direct', delivery_mode=2):
        return self.Exchange(None)

    def _create_binding(self, task_id):
        """Create new binding for task with id."""
        return self.binding

    def ensure_chords_allowed(self):
        raise NotImplementedError(E_NO_CHORD_SUPPORT.strip())

    def on_task_call(self, producer, task_id):
        if not task_join_will_block():
            maybe_declare(self.binding(producer.channel), retry=True)

    def destination_for(self, task_id, request):
        """Get the destination for result by task id.

        Returns:
            Tuple[str, str]: tuple of ``(reply_to, correlation_id)``.
        """
        try:
            request = request or current_task.request
        except AttributeError:
            raise RuntimeError(f'RPC backend missing task request for {task_id!r}')
        return (request.reply_to, request.correlation_id or task_id)

    def on_reply_declare(self, task_id):
        pass

    def on_result_fulfilled(self, result):
        pass

    def as_uri(self, include_password=True):
        return 'rpc://'

    def store_result(self, task_id, result, state, traceback=None, request=None, **kwargs):
        """Send task return value and state."""
        routing_key, correlation_id = self.destination_for(task_id, request)
        if not routing_key:
            return
        with self.app.amqp.producer_pool.acquire(block=True) as producer:
            producer.publish(self._to_result(task_id, state, result, traceback, request), exchange=self.exchange, routing_key=routing_key, correlation_id=correlation_id, serializer=self.serializer, retry=True, retry_policy=self.retry_policy, declare=self.on_reply_declare(task_id), delivery_mode=self.delivery_mode)
        return result

    def _to_result(self, task_id, state, result, traceback, request):
        return {'task_id': task_id, 'status': state, 'result': self.encode_result(result, state), 'traceback': traceback, 'children': self.current_task_children(request)}

    def on_out_of_band_result(self, task_id, message):
        if self.result_consumer:
            self.result_consumer.on_out_of_band_result(message)
        self._out_of_band[task_id] = message

    def get_task_meta(self, task_id, backlog_limit=1000):
        buffered = self._out_of_band.pop(task_id, None)
        if buffered:
            return self._set_cache_by_message(task_id, buffered)
        latest_by_id = {}
        prev = None
        for acc in self._slurp_from_queue(task_id, self.accept, backlog_limit):
            tid = self._get_message_task_id(acc)
            prev, latest_by_id[tid] = (latest_by_id.get(tid), acc)
            if prev:
                prev.ack()
                prev = None
        latest = latest_by_id.pop(task_id, None)
        for tid, msg in latest_by_id.items():
            self.on_out_of_band_result(tid, msg)
        if latest:
            latest.requeue()
            return self._set_cache_by_message(task_id, latest)
        else:
            try:
                return self._cache[task_id]
            except KeyError:
                return {'status': states.PENDING, 'result': None}
    poll = get_task_meta

    def _set_cache_by_message(self, task_id, message):
        payload = self._cache[task_id] = self.meta_from_decoded(message.payload)
        return payload

    def _slurp_from_queue(self, task_id, accept, limit=1000, no_ack=False):
        with self.app.pool.acquire_channel(block=True) as (_, channel):
            binding = self._create_binding(task_id)(channel)
            binding.declare()
            for _ in range(limit):
                msg = binding.get(accept=accept, no_ack=no_ack)
                if not msg:
                    break
                yield msg
            else:
                raise self.BacklogLimitExceeded(task_id)

    def _get_message_task_id(self, message):
        try:
            return message.properties['correlation_id']
        except (AttributeError, KeyError):
            return message.payload['task_id']

    def revive(self, channel):
        pass

    def reload_task_result(self, task_id):
        raise NotImplementedError('reload_task_result is not supported by this backend.')

    def reload_group_result(self, task_id):
        """Reload group result, even if it has been previously fetched."""
        raise NotImplementedError('reload_group_result is not supported by this backend.')

    def save_group(self, group_id, result):
        raise NotImplementedError('save_group is not supported by this backend.')

    def restore_group(self, group_id, cache=True):
        raise NotImplementedError('restore_group is not supported by this backend.')

    def delete_group(self, group_id):
        raise NotImplementedError('delete_group is not supported by this backend.')

    def __reduce__(self, args=(), kwargs=None):
        kwargs = {} if not kwargs else kwargs
        return super().__reduce__(args, dict(kwargs, connection=self._connection, exchange=self.exchange.name, exchange_type=self.exchange.type, persistent=self.persistent, serializer=self.serializer, auto_delete=self.auto_delete, expires=self.expires))

    @property
    def binding(self):
        return self.Queue(self.oid, self.exchange, self.oid, durable=False, auto_delete=True, expires=self.expires)

    @cached_property
    def oid(self):
        return self.app.thread_oid