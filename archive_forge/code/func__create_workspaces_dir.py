import atexit
import json
import os
import shutil
import sys
import tempfile
from os import path as osp
from os.path import join as pjoin
from stat import S_IRGRP, S_IROTH, S_IRUSR
from tempfile import TemporaryDirectory
from unittest.mock import patch
import jupyter_core
import jupyterlab_server
from ipykernel.kernelspec import write_kernel_spec
from jupyter_server.serverapp import ServerApp
from jupyterlab_server.process_app import ProcessApp
from traitlets import default
def _create_workspaces_dir():
    """Create a temporary directory for workspaces."""
    root_dir = tempfile.mkdtemp(prefix='mock_workspaces')
    atexit.register(lambda: shutil.rmtree(root_dir, True))
    return root_dir