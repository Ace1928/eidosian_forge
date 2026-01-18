import abc
import functools
import inspect
import logging
import threading
import traceback
from oslo_config import cfg
from oslo_service import service
from oslo_utils import eventletutils
from oslo_utils import timeutils
from stevedore import driver
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
class MessageHandlingServer(service.ServiceBase, _OrderedTaskRunner, metaclass=abc.ABCMeta):
    """Server for handling messages.

    Connect a transport to a dispatcher that knows how to process the
    message using an executor that knows how the app wants to create
    new tasks.
    """

    def __init__(self, transport, dispatcher, executor=None):
        """Construct a message handling server.

        The dispatcher parameter is a DispatcherBase instance which is used
        for routing request to endpoint for processing.

        The executor parameter controls how incoming messages will be received
        and dispatched. Executor is automatically detected from
        execution environment.
        It handles many message in parallel. If your application need
        asynchronism then you need to consider to use the eventlet executor.

        :param transport: the messaging transport
        :type transport: Transport
        :param dispatcher: has a dispatch() method which is invoked for each
                           incoming request
        :type dispatcher: DispatcherBase
        :param executor: name of message executor - available values are
                         'eventlet' and 'threading'
        :type executor: str
        """
        if executor and executor not in ('threading', 'eventlet'):
            raise ExecutorLoadFailure(executor, "Executor should be None or 'eventlet' and 'threading'")
        if not executor:
            executor = utils.get_executor_with_context()
        self.conf = transport.conf
        self.conf.register_opts(_pool_opts)
        self.transport = transport
        self.dispatcher = dispatcher
        self.executor_type = executor
        if self.executor_type == 'eventlet':
            eventletutils.warn_eventlet_not_patched(expected_patched_modules=['thread'], what="the 'oslo.messaging eventlet executor'")
        self.listener = None
        try:
            mgr = driver.DriverManager('oslo.messaging.executors', self.executor_type)
        except RuntimeError as ex:
            raise ExecutorLoadFailure(self.executor_type, ex)
        self._executor_cls = mgr.driver
        self._work_executor = None
        self._started = False
        super(MessageHandlingServer, self).__init__()

    def _on_incoming(self, incoming):
        """Handles on_incoming event

        :param incoming: incoming request.
        """
        self._work_executor.submit(self._process_incoming, incoming)

    @abc.abstractmethod
    def _process_incoming(self, incoming):
        """Perform processing incoming request

        :param incoming: incoming request.
        """

    @abc.abstractmethod
    def _create_listener(self):
        """Creates listener object for polling requests
        :return: MessageListenerAdapter
        """

    @ordered(reset_after='stop')
    def start(self, override_pool_size=None):
        """Start handling incoming messages.

        This method causes the server to begin polling the transport for
        incoming messages and passing them to the dispatcher. Message
        processing will continue until the stop() method is called.

        The executor controls how the server integrates with the applications
        I/O handling strategy - it may choose to poll for messages in a new
        process, thread or co-operatively scheduled coroutine or simply by
        registering a callback with an event loop. Similarly, the executor may
        choose to dispatch messages in a new thread, coroutine or simply the
        current thread.
        """
        if self._started:
            LOG.warning('The server has already been started. Ignoring the redundant call to start().')
            return
        self._started = True
        executor_opts = {}
        executor_opts['max_workers'] = override_pool_size or self.conf.executor_thread_pool_size
        self._work_executor = self._executor_cls(**executor_opts)
        try:
            self.listener = self._create_listener()
        except driver_base.TransportDriverError as ex:
            raise ServerListenError(self.target, ex)
        self.listener.start(self._on_incoming)

    @ordered(after='start')
    def stop(self):
        """Stop handling incoming messages.

        Once this method returns, no new incoming messages will be handled by
        the server. However, the server may still be in the process of handling
        some messages, and underlying driver resources associated to this
        server are still in use. See 'wait' for more details.
        """
        if self.listener:
            self.listener.stop()
        self._started = False

    @ordered(after='stop')
    def wait(self):
        """Wait for message processing to complete.

        After calling stop(), there may still be some existing messages
        which have not been completely processed. The wait() method blocks
        until all message processing has completed.

        Once it's finished, the underlying driver resources associated to this
        server are released (like closing useless network connections).
        """
        self._work_executor.shutdown(wait=True)
        if self.listener:
            self.listener.cleanup()

    def reset(self):
        """Reset service.

        Called in case service running in daemon mode receives SIGHUP.
        """
        pass