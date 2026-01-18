import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def _validate_sections(self, config: 'ConfigParser') -> None:
    default_section = config.defaults()
    if default_section:
        err_title = 'Found config values without a top-level section'
        err_msg = 'not part of a section'
        err = [{'loc': [k], 'msg': err_msg} for k in default_section]
        raise ConfigValidationError(errors=err, title=err_title)