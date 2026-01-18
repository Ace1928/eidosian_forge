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
class ParameterExpression(_expr_core.OperatorMixin, object):

    def __init__(self, expr) -> None:
        self.expr = expr

    def to_dict(self) -> TypingDict[str, str]:
        return {'expr': repr(self.expr)}

    def _to_expr(self) -> str:
        return repr(self.expr)

    def _from_expr(self, expr) -> 'ParameterExpression':
        return ParameterExpression(expr=expr)