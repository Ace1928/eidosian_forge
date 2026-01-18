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
class _PromiseSchemaConfig:
    extra = 'forbid'
    arbitrary_types_allowed = True
    alias_generator = alias_generator