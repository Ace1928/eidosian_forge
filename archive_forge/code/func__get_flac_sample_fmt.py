import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _get_flac_sample_fmt(bps):
    if bps is None or bps == 16:
        return 's16'
    if bps == 24:
        return 's32'
    raise ValueError(f'FLAC only supports bits_per_sample values of 16 and 24 ({bps} specified).')