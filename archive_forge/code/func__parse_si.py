from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def _parse_si(i):
    media_type = i.media_type
    if media_type == 'audio':
        return SourceAudioStream(media_type=i.media_type, codec=i.codec_name, codec_long_name=i.codec_long_name, format=i.format, bit_rate=i.bit_rate, num_frames=i.num_frames, bits_per_sample=i.bits_per_sample, metadata=i.metadata, sample_rate=i.sample_rate, num_channels=i.num_channels)
    if media_type == 'video':
        return SourceVideoStream(media_type=i.media_type, codec=i.codec_name, codec_long_name=i.codec_long_name, format=i.format, bit_rate=i.bit_rate, num_frames=i.num_frames, bits_per_sample=i.bits_per_sample, metadata=i.metadata, width=i.width, height=i.height, frame_rate=i.frame_rate)
    return SourceStream(media_type=i.media_type, codec=i.codec_name, codec_long_name=i.codec_long_name, format=None, bit_rate=None, num_frames=None, bits_per_sample=None, metadata=i.metadata)