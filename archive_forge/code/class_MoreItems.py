from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class MoreItems:

    def __init__(self, value, handled_items):
        self.value = value
        self.handled_items = handled_items

    def get_contents_debug_adapter_protocol(self, _self, fmt=None):
        total_items = len(self.value)
        remaining = total_items - self.handled_items
        bucket_size = pydevd_constants.PYDEVD_CONTAINER_BUCKET_SIZE
        from_i = self.handled_items
        to_i = from_i + min(bucket_size, remaining)
        ret = []
        while remaining > 0:
            remaining -= bucket_size
            more_items_range = MoreItemsRange(self.value, from_i, to_i)
            ret.append((str(more_items_range), more_items_range, None))
            from_i = to_i
            to_i = from_i + min(bucket_size, remaining)
        return ret

    def get_dictionary(self, _self, fmt=None):
        dct = {}
        for key, obj, _ in self.get_contents_debug_adapter_protocol(self, fmt):
            dct[key] = obj
        return dct

    def resolve(self, attribute):
        from_i, to_i = attribute[1:-1].split(':')
        from_i = int(from_i)
        to_i = int(to_i)
        return MoreItemsRange(self.value, from_i, to_i)

    def __eq__(self, o):
        return isinstance(o, MoreItems) and self.value is o.value

    def __str__(self):
        return '...'
    __repr__ = __str__