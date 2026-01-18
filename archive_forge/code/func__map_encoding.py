import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _map_encoding(encoding: str) -> str:
    for dst in ['PCM_S', 'PCM_U', 'PCM_F']:
        if dst in encoding:
            return dst
    if encoding == 'PCM_MULAW':
        return 'ULAW'
    elif encoding == 'PCM_ALAW':
        return 'ALAW'
    return encoding