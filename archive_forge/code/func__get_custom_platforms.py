import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
def _get_custom_platforms(arch: str) -> List[str]:
    arch_prefix, arch_sep, arch_suffix = arch.partition('_')
    if arch.startswith('macosx'):
        arches = _mac_platforms(arch)
    elif arch_prefix in ['manylinux2014', 'manylinux2010']:
        arches = _custom_manylinux_platforms(arch)
    else:
        arches = [arch]
    return arches