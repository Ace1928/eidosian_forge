import sys
import threading
import warnings
import weakref
from weakref import WeakMethod
from kombu.utils.functional import retry_over_time
from celery.exceptions import CDeprecationWarning
from celery.local import PromiseProxy, Proxy
from celery.utils.functional import fun_accepts_kwargs
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
def _handle_options(sender=None, weak=True, dispatch_uid=None, retry=False):

    def _connect_signal(fun):
        options = {'dispatch_uid': dispatch_uid, 'weak': weak}

        def _retry_receiver(retry_fun):

            def _try_receiver_over_time(*args, **kwargs):

                def on_error(exc, intervals, retries):
                    interval = next(intervals)
                    err_msg = RECEIVER_RETRY_ERROR % {'receiver': retry_fun, 'when': humanize_seconds(interval, 'in', ' ')}
                    logger.error(err_msg)
                    return interval
                return retry_over_time(retry_fun, Exception, args, kwargs, on_error)
            return _try_receiver_over_time
        if retry:
            options['weak'] = False
            if not dispatch_uid:
                options['dispatch_uid'] = _make_id(fun)
            fun = _retry_receiver(fun)
        self._connect_signal(fun, sender, options['weak'], options['dispatch_uid'])
        return fun
    return _connect_signal