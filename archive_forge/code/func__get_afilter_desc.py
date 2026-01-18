from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def _get_afilter_desc(sample_rate: Optional[int], fmt: Optional[str], num_channels: Optional[int]):
    descs = []
    if sample_rate is not None:
        descs.append(f'aresample={sample_rate}')
    if fmt is not None or num_channels is not None:
        parts = []
        if fmt is not None:
            parts.append(f'sample_fmts={fmt}')
        if num_channels is not None:
            parts.append(f'channel_layouts={num_channels}c')
        descs.append(f'aformat={':'.join(parts)}')
    return ','.join(descs) if descs else None