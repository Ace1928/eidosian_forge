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
class DottedOrNone(DottedObjectName):
    """A string holding a valid dotted object name in Python, such as A.b3._c
    Also allows for None type.
    """
    default_value = ''

    def validate(self, obj, value):
        """Validate an input."""
        if value is not None and len(value) > 0:
            return super().validate(obj, value)
        return value