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
def _vm_has_s3_controller(self, vm_name):
    return self.get_vm_generation(vm_name) == constants.VM_GEN_1