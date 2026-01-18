import base64
from collections import OrderedDict
import importlib
import io
import zlib
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.rllib.utils.error import NotSerializable
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.simplex import Simplex
def _serialize_ndarray(array: np.ndarray) -> str:
    """Pack numpy ndarray into Base64 encoded strings for serialization.

    This function uses numpy.save() instead of pickling to ensure
    compatibility.

    Args:
        array: numpy ndarray.

    Returns:
        b64 escaped string.
    """
    buf = io.BytesIO()
    np.save(buf, array)
    return base64.b64encode(zlib.compress(buf.getvalue())).decode('ascii')