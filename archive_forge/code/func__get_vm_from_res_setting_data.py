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
def _get_vm_from_res_setting_data(self, res_setting_data):
    vmsettings_instance_id = res_setting_data.InstanceID.split('\\')[0]
    sd = self._conn.Msvm_VirtualSystemSettingData(InstanceID=vmsettings_instance_id)
    vm = self._conn.Msvm_ComputerSystem(Name=sd[0].ConfigurationID)
    return vm[0]