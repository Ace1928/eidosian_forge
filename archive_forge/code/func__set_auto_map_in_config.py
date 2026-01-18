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
def _set_auto_map_in_config(_config):
    module_name = obj.__class__.__module__
    last_module = module_name.split('.')[-1]
    full_name = f'{last_module}.{obj.__class__.__name__}'
    if 'Tokenizer' in full_name:
        slow_tokenizer_class = None
        fast_tokenizer_class = None
        if obj.__class__.__name__.endswith('Fast'):
            fast_tokenizer_class = f'{last_module}.{obj.__class__.__name__}'
            if getattr(obj, 'slow_tokenizer_class', None) is not None:
                slow_tokenizer = getattr(obj, 'slow_tokenizer_class')
                slow_tok_module_name = slow_tokenizer.__module__
                last_slow_tok_module = slow_tok_module_name.split('.')[-1]
                slow_tokenizer_class = f'{last_slow_tok_module}.{slow_tokenizer.__name__}'
        else:
            slow_tokenizer_class = f'{last_module}.{obj.__class__.__name__}'
        full_name = (slow_tokenizer_class, fast_tokenizer_class)
    if isinstance(_config, dict):
        auto_map = _config.get('auto_map', {})
        auto_map[obj._auto_class] = full_name
        _config['auto_map'] = auto_map
    elif getattr(_config, 'auto_map', None) is not None:
        _config.auto_map[obj._auto_class] = full_name
    else:
        _config.auto_map = {obj._auto_class: full_name}