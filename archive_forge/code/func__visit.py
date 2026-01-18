import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
def _visit(feature: FeatureType, func: Callable[[FeatureType], Optional[FeatureType]]) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        visited feature (FeatureType)
    """
    if isinstance(feature, dict):
        out = func({k: _visit(f, func) for k, f in feature.items()})
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func)])
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit(feature.feature, func), length=feature.length))
    else:
        out = func(feature)
    return feature if out is None else out