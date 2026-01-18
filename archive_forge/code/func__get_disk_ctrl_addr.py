import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def _get_disk_ctrl_addr(self, controller_path):
    ctrl = self._get_wmi_obj(controller_path)
    if ctrl.ResourceSubType == self._IDE_CTRL_RES_SUB_TYPE:
        return ctrl.Address
    vmsettings = ctrl.associators(wmi_result_class=self._VIRTUAL_SYSTEM_SETTING_DATA_CLASS)[0]
    scsi_ctrls = self._get_vm_disk_controllers(vmsettings, self._SCSI_CTRL_RES_SUB_TYPE)
    ctrl_paths = [rasd.path_().upper() for rasd in scsi_ctrls]
    if controller_path.upper() in ctrl_paths:
        return ctrl_paths.index(controller_path.upper())