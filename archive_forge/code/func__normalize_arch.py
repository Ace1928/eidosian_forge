from __future__ import (absolute_import, division, print_function)
import re
def _normalize_arch(arch_str, variant_str):
    arch_str = arch_str.lower()
    variant_str = variant_str.lower()
    res = _NORMALIZE_ARCH.get((arch_str, variant_str))
    if res is None:
        res = _NORMALIZE_ARCH.get((arch_str, None))
    if res is None:
        return (arch_str, variant_str)
    if res is not None:
        arch_str = res[0]
        if res[1] is not None:
            variant_str = res[1]
        return (arch_str, variant_str)