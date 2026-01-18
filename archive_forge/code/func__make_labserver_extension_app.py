from __future__ import annotations
import json
import os
import os.path as osp
import shutil
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Callable
import pytest
from jupyter_server.serverapp import ServerApp
from jupyterlab_server import LabServerApp
def _make_labserver_extension_app(**kwargs: Any) -> LabServerApp:
    """Factory function for lab server extension apps."""
    return LabServerApp(static_dir=str(jp_root_dir), templates_dir=str(jp_template_dir), app_url='/lab', app_settings_dir=str(app_settings_dir), user_settings_dir=str(user_settings_dir), schemas_dir=str(schemas_dir), workspaces_dir=str(workspaces_dir), extra_labextensions_path=[str(labextensions_dir)])