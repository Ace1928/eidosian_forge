import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def add_dict_references():
    for key, value in obj.items():
        add_reference('key', key)
        add_reference(f'[{repr(key)}]', value)