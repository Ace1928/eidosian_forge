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
def _set_vm_vcpus(self, vmsetting, vcpus_num, vcpus_per_numa_node, limit_cpu_features):
    procsetting = _wqlutils.get_element_associated_class(self._compat_conn, self._PROCESSOR_SETTING_DATA_CLASS, element_instance_id=vmsetting.InstanceID)[0]
    vcpus = int(vcpus_num)
    procsetting.VirtualQuantity = vcpus
    procsetting.Reservation = vcpus
    procsetting.Limit = 100000
    procsetting.LimitProcessorFeatures = limit_cpu_features
    if vcpus_per_numa_node:
        procsetting.MaxProcessorsPerNumaNode = vcpus_per_numa_node
    self._jobutils.modify_virt_resource(procsetting)