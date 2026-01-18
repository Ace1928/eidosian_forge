from __future__ import annotations
import html
import json
import os
import typing as t
import uuid
import warnings
from pathlib import Path
from jinja2 import (
from jupyter_core.paths import jupyter_path
from nbformat import NotebookNode
from traitlets import Bool, Dict, HasTraits, List, Unicode, default, observe, validate
from traitlets.config import Config
from traitlets.utils.importstring import import_item
from nbconvert import filters
from .exporter import Exporter
@default('template_paths')
def _template_paths(self, prune=True, root_dirs=None):
    paths = []
    root_dirs = self.get_prefix_root_dirs()
    template_names = self.get_template_names()
    for template_name in template_names:
        for base_dir in self.extra_template_basedirs:
            path = os.path.join(base_dir, template_name)
            if not prune or os.path.exists(path):
                paths.append(path)
        for root_dir in root_dirs:
            base_dir = os.path.join(root_dir, 'nbconvert', 'templates')
            path = os.path.join(base_dir, template_name)
            if not prune or os.path.exists(path):
                paths.append(path)
    for root_dir in root_dirs:
        paths.append(root_dir)
        base_dir = os.path.join(root_dir, 'nbconvert', 'templates')
        paths.append(base_dir)
        compatibility_dir = os.path.join(root_dir, 'nbconvert', 'templates', 'compatibility')
        paths.append(compatibility_dir)
    additional_paths = []
    for path in self.template_data_paths:
        if not prune or os.path.exists(path):
            additional_paths.append(path)
    return paths + self.extra_template_paths + additional_paths