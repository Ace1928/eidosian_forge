import copy
import weakref
from pyomo.common.autoslots import AutoSlots
def _update_parent_and_storage_key(self, parent, key):
    object.__setattr__(self, '_parent', weakref.ref(parent))
    object.__setattr__(self, '_storage_key', key)