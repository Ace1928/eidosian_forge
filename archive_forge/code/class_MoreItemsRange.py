from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class MoreItemsRange:

    def __init__(self, value, from_i, to_i):
        self.value = value
        self.from_i = from_i
        self.to_i = to_i

    def get_contents_debug_adapter_protocol(self, _self, fmt=None):
        l = len(self.value)
        ret = []
        format_str = '%0' + str(int(len(str(l - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(l).lstrip('0x')))) + 'x'
        for i, item in enumerate(self.value[self.from_i:self.to_i]):
            i += self.from_i
            ret.append((format_str % i, item, '[%s]' % i))
        return ret

    def get_dictionary(self, _self, fmt=None):
        dct = {}
        for key, obj, _ in self.get_contents_debug_adapter_protocol(self, fmt):
            dct[key] = obj
        return dct

    def resolve(self, attribute):
        """
        :param var: that's the original object we're dealing with.
        :param attribute: that's the key to resolve
            -- either the dict key in get_dictionary or the name in the dap protocol.
        """
        return self.value[int(attribute)]

    def __eq__(self, o):
        return isinstance(o, MoreItemsRange) and self.value is o.value and (self.from_i == o.from_i) and (self.to_i == o.to_i)

    def __str__(self):
        return '[%s:%s]' % (self.from_i, self.to_i)
    __repr__ = __str__