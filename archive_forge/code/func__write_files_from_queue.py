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
def _write_files_from_queue(file_queue: queue.Queue, result_queue: queue.Queue, planner: SavePlanner, inflight_threshhold: int, use_fsync: bool):
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader
            if torch.cuda.is_available() and inflight_threshhold > 0:
                loader = _OverlappingCpuLoader(planner.resolve_data, inflight_threshhold=inflight_threshhold)
            else:
                loader = _SerialCpuLoader(planner.resolve_data)
            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()
            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []
            with file_name.open('wb') as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(_write_item(stream, data, write_item, storage_key))
                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(_write_item(stream, tensor, write_item, storage_key))
                if use_fsync:
                    os.fsync(stream.fileno())
            result_queue.put(write_results)
    except queue.Empty:
        pass