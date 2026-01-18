import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (

    Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder. Optionally
    adds the proper fields in a config.

    Args:
        obj (`Any`): The object for which to save the module files.
        folder (`str` or `os.PathLike`): The folder where to save.
        config (`PretrainedConfig` or dictionary, `optional`):
            A config in which to register the auto_map corresponding to this custom object.

    Returns:
        `List[str]`: The list of files saved.
    