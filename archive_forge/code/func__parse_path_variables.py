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
def _parse_path_variables(pathname, path_template):
    """
    creates the dict of path variables passed to the layout
    e.g. path_template= "/asset/<asset_id>"
         if pathname provided by the browser is "/assets/a100"
         returns **{"asset_id": "a100"}
    """
    wildcard_pattern = re.sub('<.*?>', '*', path_template)
    var_pattern = re.sub('<.*?>', '(.*)', path_template)
    if not fnmatch(pathname, wildcard_pattern):
        return None
    var_names = re.findall('<(.*?)>', path_template)
    variables = re.findall(var_pattern, pathname)
    variables = variables[0] if isinstance(variables[0], tuple) else variables
    return dict(zip(var_names, variables))