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
@validate('template_name')
def _template_name_validate(self, change):
    template_name = change['value']
    if template_name and template_name.endswith('.tpl'):
        warnings.warn(f"5.x style template name passed '{self.template_name}'. Use --template-name for the template directory with a index.<ext>.j2 file and/or --template-file to denote a different template.", DeprecationWarning, stacklevel=2)
        directory, self.template_file = os.path.split(self.template_name)
        if directory:
            directory, template_name = os.path.split(directory)
        if directory and os.path.isabs(directory):
            self.extra_template_basedirs = [directory]
    return template_name