import importlib
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from os.path import basename, normpath
from os.path import join as pjoin
from jupyter_core.paths import ENV_JUPYTER_PATH, SYSTEM_JUPYTER_PATH, jupyter_data_dir
from jupyter_core.utils import ensure_dir_exists
from jupyter_server.extension.serverextension import ArgumentConflict
from jupyterlab_server.config import get_federated_extensions
from .commands import _test_overlap
Get the list of labextension paths associated with a Python module.

    Returns a tuple of (the module path,             [{
        'src': 'mockextension',
        'dest': '_mockdestination'
    }])

    Parameters
    ----------

    module : str
        Importable Python module exposing the
        magic-named `_jupyter_labextension_paths` function
    