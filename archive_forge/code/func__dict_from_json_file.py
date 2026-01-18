import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
@classmethod
def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, 'r', encoding='utf-8') as reader:
        text = reader.read()
    return json.loads(text)