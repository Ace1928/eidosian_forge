from abc import ABC, abstractmethod
import queue
import threading
import collections
from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast
import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path
from .metadata import (
from .storage import (
from .planner import (
from .utils import _create_file_view
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch._utils import _get_device_module
def _split_by_size_and_type(bins, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]
    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]
    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]
    tensor_w.sort(key=_item_size, reverse=True)
    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)
    for wi in tensor_w:
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)
    return buckets