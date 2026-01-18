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
def _get_vswitch(self, vswitch_name):
    if vswitch_name in self._switches:
        return self._switches[vswitch_name]
    vswitch = self._conn.Msvm_VirtualEthernetSwitch(ElementName=vswitch_name)
    if not vswitch:
        raise exceptions.HyperVvSwitchNotFound(vswitch_name=vswitch_name)
    if self._enable_cache:
        self._switches[vswitch_name] = vswitch[0]
    return vswitch[0]