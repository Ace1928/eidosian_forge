import hashlib
import os
import platform
import re
import shutil
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union, cast
import wandb
from wandb import util
from wandb._globals import _datatypes_callback
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.paths import LogicalPath
from .wb_value import WBValue
def _numpy_arrays_to_lists(payload: Union[dict, Sequence, 'np.ndarray']) -> Union[Sequence, dict, str, int, float, bool]:
    if isinstance(payload, dict):
        res = {}
        for key, val in payload.items():
            res[key] = _numpy_arrays_to_lists(val)
        return res
    elif isinstance(payload, Sequence) and (not isinstance(payload, str)):
        return [_numpy_arrays_to_lists(v) for v in payload]
    elif util.is_numpy_array(payload):
        if TYPE_CHECKING:
            payload = cast('np.ndarray', payload)
        return [_numpy_arrays_to_lists(v) for v in (payload.tolist() if payload.ndim > 0 else [payload.tolist()])]
    elif isinstance(payload, Media):
        return str(payload.__class__.__name__)
    return payload