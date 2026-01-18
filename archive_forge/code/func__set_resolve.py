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
@utils.use_signature(core.Resolve)
def _set_resolve(self, **kwargs):
    """Copy the chart and update the resolve property with kwargs"""
    if not hasattr(self, 'resolve'):
        raise ValueError("{} object has no attribute 'resolve'".format(self.__class__))
    copy = self.copy(deep=['resolve'])
    if copy.resolve is Undefined:
        copy.resolve = core.Resolve()
    for key, val in kwargs.items():
        copy.resolve[key] = val
    return copy