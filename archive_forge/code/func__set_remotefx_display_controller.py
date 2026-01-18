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
def _set_remotefx_display_controller(self, vm, remotefx_disp_ctrl_res, monitor_count, max_resolution, vram_bytes=None):
    new_wmi_obj = False
    if not remotefx_disp_ctrl_res:
        new_wmi_obj = True
        remotefx_disp_ctrl_res = self._get_new_resource_setting_data(self._REMOTEFX_DISP_CTRL_RES_SUB_TYPE, self._REMOTEFX_DISP_ALLOCATION_SETTING_DATA_CLASS)
    remotefx_disp_ctrl_res.MaximumMonitors = monitor_count
    remotefx_disp_ctrl_res.MaximumScreenResolution = max_resolution
    self._set_remotefx_vram(remotefx_disp_ctrl_res, vram_bytes)
    if new_wmi_obj:
        self._jobutils.add_virt_resource(remotefx_disp_ctrl_res, vm)
    else:
        self._jobutils.modify_virt_resource(remotefx_disp_ctrl_res)