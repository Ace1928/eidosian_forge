import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
def __setup_signal_interruption(self):
    """Set up to do the Right Thing with signals during poll() and sleep().

        Deal with the changes introduced in PEP 475 that prevent a signal from
        interrupting eventlet's call to poll() or sleep().
        """
    select_module = eventlet.patcher.original('select')
    self.__force_interrupt_on_signal = hasattr(select_module, 'poll')
    if self.__force_interrupt_on_signal:
        try:
            from eventlet.hubs import poll as poll_hub
        except ImportError:
            pass
        else:

            def do_sleep(time_sleep_func, seconds):
                return time_sleep_func(seconds)
            time_sleep = eventlet.patcher.original('time').sleep

            @functools.wraps(time_sleep)
            def sleep_wrapper(seconds):
                try:
                    return do_sleep(time_sleep, seconds)
                except (IOError, InterruptedError) as err:
                    if err.errno != errno.EINTR:
                        raise
            poll_hub.sleep = sleep_wrapper
        hub = eventlet.hubs.get_hub()
        self.__hub_module_file = sys.modules[hub.__module__].__file__