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
def _init_preprocessors(self):
    super()._init_preprocessors()
    conf = self._get_conf()
    preprocessors = conf.get('preprocessors', {})
    for _, preprocessor in sorted(preprocessors.items(), key=lambda x: x[0]):
        if preprocessor is not None:
            kwargs = preprocessor.copy()
            preprocessor_cls = kwargs.pop('type')
            preprocessor_cls = import_item(preprocessor_cls)
            if preprocessor_cls.__name__ in self.config:
                kwargs.update(self.config[preprocessor_cls.__name__])
            preprocessor = preprocessor_cls(**kwargs)
            self.register_preprocessor(preprocessor)