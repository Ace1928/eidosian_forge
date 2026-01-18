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
def _create_environment(self):
    """
        Create the Jinja templating environment.
        """
    paths = self.template_paths
    self.log.debug('Template paths:\n\t%s', '\n\t'.join(paths))
    loaders = [*self.extra_loaders, ExtensionTolerantLoader(FileSystemLoader(paths), self.template_extension), DictLoader({self._raw_template_key: self.raw_template})]
    environment = Environment(loader=ChoiceLoader(loaders), extensions=JINJA_EXTENSIONS, enable_async=self.enable_async)
    environment.globals['uuid4'] = uuid.uuid4
    for key, value in self.default_filters():
        self._register_filter(environment, key, value)
    if self.filters:
        for key, user_filter in self.filters.items():
            self._register_filter(environment, key, user_filter)
    return environment