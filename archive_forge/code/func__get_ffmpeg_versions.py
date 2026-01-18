import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _get_ffmpeg_versions():
    ffmpeg_vers = _FFMPEG_VERS
    if (ffmpeg_ver := os.environ.get('TORIO_USE_FFMPEG_VERSION')) is not None:
        if ffmpeg_ver not in ffmpeg_vers:
            raise ValueError(f"The FFmpeg version '{ffmpeg_ver}' (read from TORIO_USE_FFMPEG_VERSION) is not one of supported values. Possible values are {ffmpeg_vers}")
        ffmpeg_vers = [ffmpeg_ver]
    return ffmpeg_vers