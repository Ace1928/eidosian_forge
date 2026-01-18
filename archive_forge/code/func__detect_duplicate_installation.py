import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _detect_duplicate_installation():
    if sys.version_info < (3, 8):
        return
    import importlib.metadata
    known = ['cupy', 'cupy-cuda80', 'cupy-cuda90', 'cupy-cuda91', 'cupy-cuda92', 'cupy-cuda100', 'cupy-cuda101', 'cupy-cuda102', 'cupy-cuda110', 'cupy-cuda111', 'cupy-cuda112', 'cupy-cuda113', 'cupy-cuda114', 'cupy-cuda115', 'cupy-cuda116', 'cupy-cuda117', 'cupy-cuda118', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-rocm-4-0', 'cupy-rocm-4-1', 'cupy-rocm-4-2', 'cupy-rocm-4-3', 'cupy-rocm-5-0']
    cupy_installed = [name for name in known if list(importlib.metadata.distributions(name=name))]
    if 1 < len(cupy_installed):
        cupy_packages_list = ', '.join(sorted(cupy_installed))
        warnings.warn(f'\n--------------------------------------------------------------------------------\n\n  CuPy may not function correctly because multiple CuPy packages are installed\n  in your environment:\n\n    {cupy_packages_list}\n\n  Follow these steps to resolve this issue:\n\n    1. For all packages listed above, run the following command to remove all\n       existing CuPy installations:\n\n         $ pip uninstall <package_name>\n\n      If you previously installed CuPy via conda, also run the following:\n\n         $ conda uninstall cupy\n\n    2. Install the appropriate CuPy package.\n       Refer to the Installation Guide for detailed instructions.\n\n         https://docs.cupy.dev/en/stable/install.html\n\n--------------------------------------------------------------------------------\n')