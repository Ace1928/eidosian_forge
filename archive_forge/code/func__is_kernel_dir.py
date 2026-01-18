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
def _is_kernel_dir(path: str) -> bool:
    """Is ``path`` a kernel directory?"""
    return os.path.isdir(path) and os.path.isfile(pjoin(path, 'kernel.json'))