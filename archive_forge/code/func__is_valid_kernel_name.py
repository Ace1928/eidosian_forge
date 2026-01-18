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
def _is_valid_kernel_name(name: str) -> t.Any:
    """Check that a kernel name is valid."""
    return _kernel_name_pat.match(name)