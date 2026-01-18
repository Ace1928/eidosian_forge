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
def _set_vm_memory(self, vmsetting, memory_mb, memory_per_numa_node, dynamic_memory_ratio):
    mem_settings = self._get_vm_memory(vmsetting)
    max_mem = int(memory_mb)
    mem_settings.Limit = max_mem
    if dynamic_memory_ratio > 1:
        mem_settings.DynamicMemoryEnabled = True
        reserved_mem = min(int(max_mem / dynamic_memory_ratio) >> 1 << 1, max_mem)
    else:
        mem_settings.DynamicMemoryEnabled = False
        reserved_mem = max_mem
    mem_settings.Reservation = reserved_mem
    mem_settings.VirtualQuantity = reserved_mem
    if memory_per_numa_node:
        mem_settings.MaxMemoryBlocksPerNumaNode = memory_per_numa_node
    self._jobutils.modify_virt_resource(mem_settings)