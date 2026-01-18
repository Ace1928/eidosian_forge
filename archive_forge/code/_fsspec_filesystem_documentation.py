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

        Initialize the writer pointing to `path`.

        Args:
            path: diretory where the checkpoint will be writen to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        