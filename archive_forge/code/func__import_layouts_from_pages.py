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
def _import_layouts_from_pages(pages_folder):
    for root, dirs, files in os.walk(pages_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.') and (not d.startswith('_'))]
        for file in files:
            if file.startswith('_') or file.startswith('.') or (not file.endswith('.py')):
                continue
            page_path = os.path.join(root, file)
            with open(page_path, encoding='utf-8') as f:
                content = f.read()
                if 'register_page' not in content:
                    continue
            module_name = _infer_module_name(page_path)
            spec = importlib.util.spec_from_file_location(module_name, page_path)
            page_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(page_module)
            sys.modules[module_name] = page_module
            if module_name in PAGE_REGISTRY and (not PAGE_REGISTRY[module_name]['supplied_layout']):
                _validate.validate_pages_layout(module_name, page_module)
                PAGE_REGISTRY[module_name]['layout'] = getattr(page_module, 'layout')