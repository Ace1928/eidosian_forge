from __future__ import annotations
import asyncio
import glob
import logging
import os
import sys
import typing as t
from textwrap import dedent, fill
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, DottedObjectName, Instance, List, Type, Unicode, default, observe
from traitlets.config import Configurable, catch_config_error
from traitlets.utils.importstring import import_item
from nbconvert import __version__, exporters, postprocessors, preprocessors, writers
from nbconvert.utils.text import indent
from .exporters.base import get_export_names, get_exporter
from .utils.base import NbConvertBase
from .utils.exceptions import ConversionException
from .utils.io import unicode_stdin_stream
class DejavuApp(NbConvertApp):
    """A deja vu app."""

    def initialize(self, argv=None):
        """Initialize the app."""
        self.config.TemplateExporter.exclude_input = True
        self.config.TemplateExporter.exclude_output_prompt = True
        self.config.TemplateExporter.exclude_input_prompt = True
        self.config.ExecutePreprocessor.enabled = True
        self.config.WebPDFExporter.paginate = False
        self.config.QtPDFExporter.paginate = False
        super().initialize(argv)
        if hasattr(self, 'load_config_environ'):
            self.load_config_environ()

    @default('export_format')
    def _default_export_format(self):
        return 'html'