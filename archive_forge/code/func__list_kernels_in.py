from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def _list_kernels_in(dir: str | None) -> dict[str, str]:
    """Return a mapping of kernel names to resource directories from dir.

    If dir is None or does not exist, returns an empty dict.
    """
    if dir is None or not os.path.isdir(dir):
        return {}
    kernels = {}
    for f in os.listdir(dir):
        path = pjoin(dir, f)
        if not _is_kernel_dir(path):
            continue
        key = f.lower()
        if not _is_valid_kernel_name(key):
            warnings.warn(f'Invalid kernelspec directory name ({_kernel_name_description}): {path}', stacklevel=3)
        kernels[key] = path
    return kernels