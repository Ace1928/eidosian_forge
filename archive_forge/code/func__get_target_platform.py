import os
import platform
from pathlib import Path
from cffi import FFI
def _get_target_platform(arch_flags, default):
    flags = [f for f in arch_flags.split(' ') if f.strip() != '']
    try:
        pos = flags.index('-arch')
        return flags[pos + 1].lower()
    except ValueError:
        pass
    return default