import functools
import random
import sys
import time
from eventlet import event
from eventlet import greenthread
from oslo_log import log as logging
from oslo_utils import eventletutils
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_service._i18n import _
class RetryDecorator(object):
    """Decorator for retrying a function upon suggested exceptions.

    The decorated function is retried for the given number of times, and the
    sleep time between the retries is incremented until max sleep time is
    reached. If the max retry count is set to -1, then the decorated function
    is invoked indefinitely until an exception is thrown, and the caught
    exception is not in the list of suggested exceptions.
    """

    def __init__(self, max_retry_count=-1, inc_sleep_time=10, max_sleep_time=60, exceptions=()):
        """Configure the retry object using the input params.

        :param max_retry_count: maximum number of times the given function must
                                be retried when one of the input 'exceptions'
                                is caught. When set to -1, it will be retried
                                indefinitely until an exception is thrown
                                and the caught exception is not in param
                                exceptions.
        :param inc_sleep_time: incremental time in seconds for sleep time
                               between retries
        :param max_sleep_time: max sleep time in seconds beyond which the sleep
                               time will not be incremented using param
                               inc_sleep_time. On reaching this threshold,
                               max_sleep_time will be used as the sleep time.
        :param exceptions: suggested exceptions for which the function must be
                           retried, if no exceptions are provided (the default)
                           then all exceptions will be reraised, and no
                           retrying will be triggered.
        """
        self._max_retry_count = max_retry_count
        self._inc_sleep_time = inc_sleep_time
        self._max_sleep_time = max_sleep_time
        self._exceptions = exceptions
        self._retry_count = 0
        self._sleep_time = 0

    def __call__(self, f):
        func_name = reflection.get_callable_name(f)

        def _func(*args, **kwargs):
            result = None
            try:
                if self._retry_count:
                    LOG.debug('Invoking %(func_name)s; retry count is %(retry_count)d.', {'func_name': func_name, 'retry_count': self._retry_count})
                result = f(*args, **kwargs)
            except self._exceptions:
                with excutils.save_and_reraise_exception() as ctxt:
                    LOG.debug('Exception which is in the suggested list of exceptions occurred while invoking function: %s.', func_name)
                    if self._max_retry_count != -1 and self._retry_count >= self._max_retry_count:
                        LOG.debug('Cannot retry %(func_name)s upon suggested exception since retry count (%(retry_count)d) reached max retry count (%(max_retry_count)d).', {'retry_count': self._retry_count, 'max_retry_count': self._max_retry_count, 'func_name': func_name})
                    else:
                        ctxt.reraise = False
                        self._retry_count += 1
                        self._sleep_time += self._inc_sleep_time
                        return self._sleep_time
            raise LoopingCallDone(result)

        @functools.wraps(f)
        def func(*args, **kwargs):
            loop = DynamicLoopingCall(_func, *args, **kwargs)
            evt = loop.start(periodic_interval_max=self._max_sleep_time)
            LOG.debug('Waiting for function %s to return.', func_name)
            return evt.wait()
        return func