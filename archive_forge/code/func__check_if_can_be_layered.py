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
def _check_if_can_be_layered(spec: Union[dict, core.SchemaBase]) -> None:
    """Check if the spec can be layered."""

    def _get(spec, attr):
        if isinstance(spec, core.SchemaBase):
            return spec._get(attr)
        else:
            return spec.get(attr, Undefined)
    encoding = _get(spec, 'encoding')
    if encoding is not Undefined:
        for channel in ['row', 'column', 'facet']:
            if _get(encoding, channel) is not Undefined:
                raise ValueError('Faceted charts cannot be layered. Instead, layer the charts before faceting.')
    if isinstance(spec, (Chart, LayerChart)):
        return
    if not isinstance(spec, (core.SchemaBase, dict)):
        raise ValueError('Only chart objects can be layered.')
    if _get(spec, 'facet') is not Undefined:
        raise ValueError('Faceted charts cannot be layered. Instead, layer the charts before faceting.')
    if isinstance(spec, FacetChart) or _get(spec, 'facet') is not Undefined:
        raise ValueError('Faceted charts cannot be layered. Instead, layer the charts before faceting.')
    if isinstance(spec, RepeatChart) or _get(spec, 'repeat') is not Undefined:
        raise ValueError('Repeat charts cannot be layered. Instead, layer the charts before repeating.')
    if isinstance(spec, ConcatChart) or _get(spec, 'concat') is not Undefined:
        raise ValueError('Concatenated charts cannot be layered. Instead, layer the charts before concatenating.')
    if isinstance(spec, HConcatChart) or _get(spec, 'hconcat') is not Undefined:
        raise ValueError('Concatenated charts cannot be layered. Instead, layer the charts before concatenating.')
    if isinstance(spec, VConcatChart) or _get(spec, 'vconcat') is not Undefined:
        raise ValueError('Concatenated charts cannot be layered. Instead, layer the charts before concatenating.')