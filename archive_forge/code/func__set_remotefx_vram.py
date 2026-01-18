import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def _set_remotefx_vram(self, remotefx_disp_ctrl_res, vram_bytes):
    if vram_bytes:
        remotefx_disp_ctrl_res.VRAMSizeBytes = six.text_type(vram_bytes)