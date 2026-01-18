from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
def _get_custom_models(models: Sequence[type[HasProps]] | None) -> dict[str, CustomModel] | None:
    """Returns CustomModels for models with a custom `__implementation__`"""
    custom_models: dict[str, CustomModel] = dict()
    for cls in models or HasProps.model_class_reverse_map.values():
        impl = getattr(cls, '__implementation__', None)
        if impl is not None:
            model = CustomModel(cls)
            custom_models[model.full_name] = model
    return custom_models if custom_models else None