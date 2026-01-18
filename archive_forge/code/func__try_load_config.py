from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def _try_load_config(self, locations: Sequence[Path]) -> dict[str, Any]:
    for location in locations:
        try:
            with location.open() as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        except Exception:
            pass
    return {}