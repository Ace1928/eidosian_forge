import collections
import dataclasses
import io
import os
import pickle
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, cast, Dict, List, Optional, Union
import fsspec
import torch
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from torch import Tensor
from torch._utils import _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.storage import (
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future
class FsspecReader(StorageReader):

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__()
        self.path = path
        self.fs, _ = url_to_fs(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

    def _slice_file(self, file, sinfo: _StorageInfo):
        return _create_file_view(file, sinfo.offset, sinfo.length)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)
        for relative_path, reqs in per_file.items():
            abs_path = os.path.join(self.path, relative_path)
            with fsspec.open(abs_path, 'rb') as file:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = io.BytesIO(file_slice.read(item_md.length))
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
                    else:
                        tensor = cast(Tensor, torch.load(file_slice, map_location='cpu'))
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()
                        assert target_tensor.size() == tensor.size(), f'req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}'
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)
        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        metadata_path = os.path.join(self.path, '.metadata')
        with fsspec.open(metadata_path, 'rb') as metadata_file:
            return pickle.load(metadata_file)

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan