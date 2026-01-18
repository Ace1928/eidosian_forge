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
def _get_vnic_settings(self, vnic_name):
    vnic_settings = self._conn.Msvm_SyntheticEthernetPortSettingData(ElementName=vnic_name)
    if not vnic_settings:
        raise exceptions.HyperVvNicNotFound(vnic_name=vnic_name)
    return vnic_settings[0]