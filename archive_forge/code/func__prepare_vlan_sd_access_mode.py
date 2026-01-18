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
def _prepare_vlan_sd_access_mode(self, vlan_settings, vlan_id):
    if vlan_settings:
        vlan_id = vlan_id or vlan_settings.AccessVlanId
        if vlan_settings.OperationMode == constants.VLAN_MODE_ACCESS and vlan_settings.AccessVlanId == vlan_id:
            return None
    vlan_settings = self._create_default_setting_data(self._PORT_VLAN_SET_DATA)
    vlan_settings.AccessVlanId = vlan_id
    vlan_settings.OperationMode = constants.VLAN_MODE_ACCESS
    return vlan_settings