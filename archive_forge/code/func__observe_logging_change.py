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
@observe('log_datefmt', 'log_format', 'log_level', 'logging_config')
def _observe_logging_change(self, change: Bunch) -> None:
    log_level = self.log_level
    if isinstance(log_level, str):
        self.log_level = t.cast(int, getattr(logging, log_level))
    self._configure_logging()