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
def _get_vm_ide_controller(self, vmsettings, ctrller_addr):
    ide_ctrls = self._get_vm_disk_controllers(vmsettings, self._IDE_CTRL_RES_SUB_TYPE)
    ctrl = [r for r in ide_ctrls if r.Address == str(ctrller_addr)]
    return ctrl[0].path_() if ctrl else None