import collections.abc as collections
import json
import os
import warnings
from functools import wraps
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
from .utils._typing import CallableT
def _requires_keras_2_model(fn: CallableT) -> CallableT:

    @wraps(fn)
    def _inner(model, *args, **kwargs):
        if not hasattr(model, 'history'):
            raise NotImplementedError(f"Cannot use '{fn.__name__}': Keras 3.x is not supported. Please save models manually and upload them using `upload_folder` or `huggingface-cli upload`.")
        return fn(model, *args, **kwargs)
    return _inner