import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def _readenv(name, ctor, default):
    value = environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        warnings.warn(f"Environment variable '{name}' is defined but its associated value '{value}' could not be parsed.\nThe parse failed with exception:\n{traceback.format_exc()}", RuntimeWarning)
        return default