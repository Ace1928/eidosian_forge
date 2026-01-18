import ast
import sys
from typing import Any
from .internal import Filters, Key
def _server_path_to_key(path):
    if path.startswith('config.'):
        return Key(section='config', name=path.split('config.', 1)[1])
    elif path.startswith('summary_metrics.'):
        return Key(section='summary', name=path.split('summary_metrics.', 1)[1])
    elif path.startswith('keys_info.keys.'):
        return Key(section='keys_info', name=path.split('keys_info.keys.', 1)[1])
    elif path.startswith('tags.'):
        return Key(section='tags', name=path.split('tags.', 1)[1])
    else:
        return Key(section='run', name=path)