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
def _get_attached_disks_query_string(self, scsi_controller_path):
    return "SELECT * FROM Msvm_ResourceAllocationSettingData WHERE (ResourceSubType='%(res_sub_type)s' OR ResourceSubType='%(res_sub_type_virt)s' OR ResourceSubType='%(res_sub_type_dvd)s') AND Parent = '%(parent)s'" % {'res_sub_type': self._PHYS_DISK_RES_SUB_TYPE, 'res_sub_type_virt': self._DISK_DRIVE_RES_SUB_TYPE, 'res_sub_type_dvd': self._DVD_DRIVE_RES_SUB_TYPE, 'parent': scsi_controller_path.replace("'", "''")}