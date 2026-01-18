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
def _box(d: Dict) -> gym.spaces.Box:
    ret = d.copy()
    ret.update({'low': _deserialize_ndarray(d['low']), 'high': _deserialize_ndarray(d['high'])})
    return gym.spaces.Box(**__common(ret))