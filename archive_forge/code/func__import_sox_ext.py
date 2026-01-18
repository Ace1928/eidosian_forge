import importlib
import logging
import os
import types
from pathlib import Path
import torch
from torchaudio._internal.module_utils import eval_env
def _import_sox_ext():
    if os.name == 'nt':
        raise RuntimeError('sox extension is not supported on Windows')
    if not eval_env('TORCHAUDIO_USE_SOX', True):
        raise RuntimeError('sox extension is disabled. (TORCHAUDIO_USE_SOX=0)')
    ext = 'torchaudio.lib._torchaudio_sox'
    if not importlib.util.find_spec(ext):
        raise RuntimeError('TorchAudio is not built with sox extension. Please build TorchAudio with libsox support. (BUILD_SOX=1)')
    _load_lib('libtorchaudio_sox')
    return importlib.import_module(ext)