import collections
import importlib
import os
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
from os.path import isfile, join
from urllib.parse import parse_qs
import flask
from . import _validate
from ._utils import AttributeDict
from ._get_paths import get_relative_path
from ._callback_context import context_value
from ._get_app import get_app
def _infer_module_name(page_path):
    relative_path = page_path.split(CONFIG.pages_folder)[-1]
    module = _path_to_module_name(relative_path)
    proj_root = flask.helpers.get_root_path(CONFIG.name)
    if CONFIG.pages_folder.startswith(proj_root):
        parent_path = CONFIG.pages_folder[len(proj_root):]
    else:
        parent_path = CONFIG.pages_folder
    parent_module = _path_to_module_name(parent_path)
    module_name = f'{parent_module}.{module}'
    if _module_name_is_package(CONFIG.name):
        module_name = f'{CONFIG.name}.{module_name}'
    return module_name