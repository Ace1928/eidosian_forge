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
def _get_destination_dir(self, kernel_name: str, user: bool=False, prefix: str | None=None) -> str:
    if user:
        return os.path.join(self.user_kernel_dir, kernel_name)
    elif prefix:
        return os.path.join(os.path.abspath(prefix), 'share', 'jupyter', 'kernels', kernel_name)
    else:
        return os.path.join(SYSTEM_JUPYTER_PATH[0], 'kernels', kernel_name)