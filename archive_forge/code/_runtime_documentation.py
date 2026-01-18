import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
Dump information about the machine to help debugging issues.

    Similar helper exist in:
    - `datasets` (https://github.com/huggingface/datasets/blob/main/src/datasets/commands/env.py)
    - `diffusers` (https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/env.py)
    - `transformers` (https://github.com/huggingface/transformers/blob/main/src/transformers/commands/env.py)
    