from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
class LevelFormatter(logging.Formatter):
    """Formatter with additional `highlevel` record

    This field is empty if log level is less than highlevel_limit,
    otherwise it is formatted with self.highlevel_format.

    Useful for adding 'WARNING' to warning messages,
    without adding 'INFO' to info, etc.
    """
    highlevel_limit = logging.WARN
    highlevel_format = ' %(levelname)s |'

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ''
        return super().format(record)