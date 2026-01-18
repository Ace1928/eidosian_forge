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
def _lookup_vm_check(self, vm_name, as_vssd=True, for_update=False):
    vm = self._lookup_vm(vm_name, as_vssd, for_update)
    if not vm:
        raise exceptions.HyperVVMNotFoundException(vm_name=vm_name)
    return vm