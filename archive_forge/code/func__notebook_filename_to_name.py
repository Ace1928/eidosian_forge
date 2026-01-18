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
def _notebook_filename_to_name(self, notebook_filename):
    """
        Returns the notebook name from the notebook filename by
        applying `output_base` pattern and stripping extension
        """
    basename = os.path.basename(notebook_filename)
    notebook_name = basename[:basename.rfind('.')]
    notebook_name = self.output_base.format(notebook_name=notebook_name)
    return notebook_name