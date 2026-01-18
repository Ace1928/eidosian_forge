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
def _get_vlan_setting_data_from_port_alloc(self, port_alloc):
    return self._get_setting_data_from_port_alloc(port_alloc, self._vlan_sds, self._PORT_VLAN_SET_DATA)