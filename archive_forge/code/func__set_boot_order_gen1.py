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
def _set_boot_order_gen1(self, vm_name, device_boot_order):
    vssd = self._lookup_vm_check(vm_name, for_update=True)
    vssd.BootOrder = tuple(device_boot_order)
    self._modify_virtual_system(vssd)