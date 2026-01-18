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
def _get_switch_port_allocation(self, switch_port_name, create=False, expected=True):
    if switch_port_name in self._switch_ports:
        return (self._switch_ports[switch_port_name], True)
    switch_port, found = self._get_setting_data(self._PORT_ALLOC_SET_DATA, switch_port_name, create)
    if found:
        if self._enable_cache:
            self._switch_ports[switch_port_name] = switch_port
    elif expected:
        raise exceptions.HyperVPortNotFoundException(port_name=switch_port_name)
    return (switch_port, found)