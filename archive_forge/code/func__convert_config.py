from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
def _convert_config(cfg: CodecConfig):
    if cfg is None:
        return None
    return ffmpeg_ext.CodecConfig(cfg.bit_rate, cfg.compression_level, cfg.qscale, cfg.gop_size, cfg.max_b_frames)