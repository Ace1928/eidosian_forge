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
class FixedIntervalWithTimeoutLoopingCall(LoopingCallBase):
    """A fixed interval looping call with timeout checking mechanism."""
    _RUN_ONLY_ONE_MESSAGE = _('A fixed interval looping call with timeout checking and can only run one function at at a time')
    _KIND = _('Fixed interval looping call with timeout checking.')

    def start(self, interval, initial_delay=None, stop_on_exception=True, timeout=0):
        start_time = time.time()

        def _idle_for(result, elapsed):
            delay = round(elapsed - interval, 2)
            if delay > 0:
                func_name = reflection.get_callable_name(self.f)
                LOG.warning('Function %(func_name)r run outlasted interval by %(delay).2f sec', {'func_name': func_name, 'delay': delay})
            elapsed_time = time.time() - start_time
            if timeout > 0 and elapsed_time > timeout:
                raise LoopingCallTimeOut(_('Looping call timed out after %.02f seconds') % elapsed_time)
            return -delay if delay < 0 else 0
        return self._start(_idle_for, initial_delay=initial_delay, stop_on_exception=stop_on_exception)