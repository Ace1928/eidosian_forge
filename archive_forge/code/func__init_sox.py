import importlib
import logging
import os
import types
from pathlib import Path
import torch
from torchaudio._internal.module_utils import eval_env
def _init_sox():
    ext = _import_sox_ext()
    ext.set_verbosity(0)
    import atexit
    torch.ops.torchaudio_sox.initialize_sox_effects()
    atexit.register(torch.ops.torchaudio_sox.shutdown_sox_effects)
    keys = ['get_info', 'load_audio_file', 'save_audio_file', 'apply_effects_tensor', 'apply_effects_file']
    for key in keys:
        setattr(ext, key, getattr(torch.ops.torchaudio_sox, key))
    return ext