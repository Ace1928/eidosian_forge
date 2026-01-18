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
def _get_vm_disk_controllers(self, vmsettings, ctrl_res_sub_type):
    rasds = _wqlutils.get_element_associated_class(self._conn, self._RESOURCE_ALLOC_SETTING_DATA_CLASS, element_instance_id=vmsettings.InstanceID)
    res = [r for r in rasds if r.ResourceSubType == ctrl_res_sub_type]
    return res