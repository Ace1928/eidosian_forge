import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def add_dynamic_timer(self, callback, initial_delay=None, periodic_interval_max=None, *args, **kwargs):
    """Add a timer that controls its own period dynamically.

        The period of each iteration of the timer is controlled by the return
        value of the callback function on the previous iteration.

        .. warning::
            Passing arguments to the callback function is deprecated. Use the
            :func:`add_dynamic_timer_args` method to pass arguments for the
            callback function.

        :param callback: The callback function to run when the timer is
                         triggered.
        :param initial_delay: The delay in seconds before first triggering the
                              timer. If not set, the timer is liable to be
                              scheduled immediately.
        :param periodic_interval_max: The maximum interval in seconds to allow
                                      the callback function to request. If
                                      provided, this is also used as the
                                      default delay if None is returned by the
                                      callback function.
        :returns: an :class:`oslo_service.loopingcall.DynamicLoopingCall`
                  instance
        """
    if args or kwargs:
        warnings.warn('Calling add_dynamic_timer() with arguments to the callback function is deprecated. Use add_dynamic_timer_args() instead.', DeprecationWarning)
    return self.add_dynamic_timer_args(callback, args, kwargs, initial_delay=initial_delay, periodic_interval_max=periodic_interval_max)