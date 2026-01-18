import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _poll_events(callback):
    if patcher.is_monkey_patched('thread'):
        listen = functools.partial(tpool.execute, listener, self._VNIC_LISTENER_TIMEOUT_MS)
    else:
        listen = functools.partial(listener, self._VNIC_LISTENER_TIMEOUT_MS)
    while True:
        try:
            event = listen()
            if event.ElementName:
                callback(event.ElementName)
            else:
                LOG.warning('Ignoring port event. The port name is missing.')
        except exceptions.x_wmi_timed_out:
            pass