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
def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool, optimize_list_casting: bool) -> Tuple[Any, bool]:
    """
    Cast pytorch/tensorflow/pandas objects to python numpy array/lists.
    It works recursively.

    If `optimize_list_casting` is True, to avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast.
        only_1d_for_numpy (bool): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
        has_changed (bool): True if the object has been changed, False if it is identical
    """
    if config.TF_AVAILABLE and 'tensorflow' in sys.modules:
        import tensorflow as tf
    if config.TORCH_AVAILABLE and 'torch' in sys.modules:
        import torch
    if config.JAX_AVAILABLE and 'jax' in sys.modules:
        import jax.numpy as jnp
    if config.PIL_AVAILABLE and 'PIL' in sys.modules:
        import PIL.Image
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return (obj[()], True)
        elif not only_1d_for_numpy or obj.ndim == 1:
            return (obj, False)
        else:
            return ([_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in obj], True)
    elif config.TORCH_AVAILABLE and 'torch' in sys.modules and isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return (obj.detach().cpu().numpy()[()], True)
        elif not only_1d_for_numpy or obj.ndim == 1:
            return (obj.detach().cpu().numpy(), True)
        else:
            return ([_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in obj.detach().cpu().numpy()], True)
    elif config.TF_AVAILABLE and 'tensorflow' in sys.modules and isinstance(obj, tf.Tensor):
        if obj.ndim == 0:
            return (obj.numpy()[()], True)
        elif not only_1d_for_numpy or obj.ndim == 1:
            return (obj.numpy(), True)
        else:
            return ([_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in obj.numpy()], True)
    elif config.JAX_AVAILABLE and 'jax' in sys.modules and isinstance(obj, jnp.ndarray):
        if obj.ndim == 0:
            return (np.asarray(obj)[()], True)
        elif not only_1d_for_numpy or obj.ndim == 1:
            return (np.asarray(obj), True)
        else:
            return ([_cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for x in np.asarray(obj)], True)
    elif config.PIL_AVAILABLE and 'PIL' in sys.modules and isinstance(obj, PIL.Image.Image):
        return (encode_pil_image(obj), True)
    elif isinstance(obj, pd.Series):
        return (_cast_to_python_objects(obj.tolist(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0], True)
    elif isinstance(obj, pd.DataFrame):
        return ({key: _cast_to_python_objects(value, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for key, value in obj.to_dict('series').items()}, True)
    elif isinstance(obj, pd.Timestamp):
        return (obj.to_pydatetime(), True)
    elif isinstance(obj, pd.Timedelta):
        return (obj.to_pytimedelta(), True)
    elif isinstance(obj, Mapping):
        has_changed = not isinstance(obj, dict)
        output = {}
        for k, v in obj.items():
            casted_v, has_changed_v = _cast_to_python_objects(v, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)
            has_changed |= has_changed_v
            output[k] = casted_v
        return (output if has_changed else obj, has_changed)
    elif hasattr(obj, '__array__'):
        return (_cast_to_python_objects(obj.__array__(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0], True)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for first_elmt in obj:
                if _check_non_null_non_empty_recursive(first_elmt):
                    break
            casted_first_elmt, has_changed_first_elmt = _cast_to_python_objects(first_elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)
            if has_changed_first_elmt or not optimize_list_casting:
                return ([_cast_to_python_objects(elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0] for elmt in obj], True)
            elif isinstance(obj, (list, tuple)):
                return (obj, False)
            else:
                return (list(obj), True)
        else:
            return (obj, False)
    else:
        return (obj, False)