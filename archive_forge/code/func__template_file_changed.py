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
@observe('template_file')
def _template_file_changed(self, change):
    new = change['new']
    if new == 'default':
        self.template_file = self.default_template
        return
    full_path = os.path.abspath(new)
    if os.path.isfile(full_path):
        directory, self.template_file = os.path.split(full_path)
        self.extra_template_paths = [directory, *self.extra_template_paths]
        if self.template_file and self.template_file.endswith('.tpl'):
            warnings.warn(f"5.x style template file passed '{new}'. Use --template-name for the template directory with a index.<ext>.j2 file and/or --template-file to denote a different template.", DeprecationWarning, stacklevel=2)