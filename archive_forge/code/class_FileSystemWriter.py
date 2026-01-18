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
class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(self, path: Union[str, os.PathLike], single_file_per_rank: bool=True, sync_files: bool=True, thread_count: int=1, per_thread_copy_ahead: int=10000000) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.path.mkdir(parents=True, exist_ok=True)
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        new_plans = [dataclasses.replace(plan, storage_data=_StoragePrefix(f'__{i}_')) for i, plan in enumerate(global_plan)]
        return new_plans

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f'{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}'
            file_count += 1
            return file_name
        file_queue: queue.Queue = queue.Queue()
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                file_queue.put((self.path / file_name, file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                file_queue.put((self.path / file_name, file_name, [item]))
        result_queue: queue.Queue = queue.Queue()
        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(target=_write_files_from_queue, args=(file_queue, result_queue, planner, self.per_thread_copy_ahead, self.sync_files))
            t.start()
            threads.append(t)
        _write_files_from_queue(file_queue=file_queue, result_queue=result_queue, planner=planner, inflight_threshhold=self.per_thread_copy_ahead, use_fsync=self.sync_files)
        for t in threads:
            t.join()
        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            pass
            fut: Future[List[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        with (self.path / '.metadata.tmp').open('wb') as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                os.fsync(metadata_file.fileno())
        (self.path / '.metadata.tmp').rename(self.path / '.metadata')