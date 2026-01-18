import logging
import sys
import warnings
from typing import Optional
import wandb
def _get_python_type() -> PythonType:
    if 'IPython' not in sys.modules:
        return 'python'
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return 'python'
    except ImportError:
        return 'python'
    ip_kernel_app_connection_file = (get_ipython().config.get('IPKernelApp', {}) or {}).get('connection_file', '').lower() or (get_ipython().config.get('ColabKernelApp', {}) or {}).get('connection_file', '').lower()
    if 'terminal' in get_ipython().__module__ or 'jupyter' not in ip_kernel_app_connection_file or 'spyder' in sys.modules:
        return 'ipython'
    else:
        return 'jupyter'