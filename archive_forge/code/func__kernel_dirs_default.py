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
def _kernel_dirs_default(self) -> list[str]:
    dirs = jupyter_path('kernels')
    try:
        from IPython.paths import get_ipython_dir
        dirs.append(os.path.join(get_ipython_dir(), 'kernels'))
    except ModuleNotFoundError:
        pass
    return dirs