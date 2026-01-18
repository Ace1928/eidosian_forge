import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _get_json_data(name: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(get_cupy_install_path(), 'cupy', '.data', name)
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)