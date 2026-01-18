import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def _summary_args(self):
    """returns a sorted sequence of 2-tuple containing the
        ``(flag_name, flag_value)`` for flag that are set with a non-default
        value.
        """
    args = []
    for k in sorted(self.options):
        opt = self.options[k]
        if self.is_set(k):
            flagval = getattr(self, k)
            if opt.default != flagval:
                v = (k, flagval)
                args.append(v)
    return args