import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
def _get_repeat_strings(repeat):
    if isinstance(repeat, list):
        return repeat
    elif isinstance(repeat, core.LayerRepeatMapping):
        klist = ['row', 'column', 'layer']
    elif isinstance(repeat, core.RepeatMapping):
        klist = ['row', 'column']
    rclist = [k for k in klist if repeat[k] is not Undefined]
    rcstrings = [[f'{k}_{v}' for v in repeat[k]] for k in rclist]
    return [''.join(s) for s in itertools.product(*rcstrings)]