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
def _path_to_page(path_id):
    path_variables = None
    for page in PAGE_REGISTRY.values():
        if page['path_template']:
            template_id = page['path_template'].strip('/')
            path_variables = _parse_path_variables(path_id, template_id)
            if path_variables:
                return (page, path_variables)
        if path_id == page['path'].strip('/'):
            return (page, path_variables)
    return ({}, None)