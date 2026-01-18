import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _preload_warning(lib, exc):
    config = get_preload_config()
    if config is not None and lib in config:
        msg = '\n{lib} library could not be loaded.\n\nReason: {exc_type} ({exc})\n\nYou can install the library by:\n'
        if config['packaging'] == 'pip':
            msg += '\n  $ python -m cupyx.tools.install_library --library {lib} --cuda {cuda}\n'
        elif config['packaging'] == 'conda':
            msg += '\n  $ conda install -c conda-forge {lib}\n'
        else:
            raise AssertionError
        msg = msg.format(lib=lib, exc_type=type(exc).__name__, exc=str(exc), cuda=config['cuda'])
        warnings.warn(msg)