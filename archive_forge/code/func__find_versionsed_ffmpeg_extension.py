import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _find_versionsed_ffmpeg_extension(version: str):
    ext = f'torio.lib._torio_ffmpeg{version}'
    lib = f'libtorio_ffmpeg{version}'
    if not importlib.util.find_spec(ext):
        raise RuntimeError(f'FFmpeg{version} extension is not available.')
    _load_lib(lib)
    return importlib.import_module(ext)