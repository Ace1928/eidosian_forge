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
def _get_setting_data_from_port_alloc(self, port_alloc, cache, data_class):
    if port_alloc.InstanceID in cache:
        return cache[port_alloc.InstanceID]
    setting_data = self._get_first_item(_wqlutils.get_element_associated_class(self._conn, data_class, element_instance_id=port_alloc.InstanceID))
    if setting_data and self._enable_cache:
        cache[port_alloc.InstanceID] = setting_data
    return setting_data