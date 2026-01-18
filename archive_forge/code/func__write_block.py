import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def _write_block(self, f: 'pyarrow.NativeFile', block: BlockAccessor, writer_args_fn: Callable[[], Dict[str, Any]]=lambda: {}, **writer_args):
    """Writes a block to a single file, passing all kwargs to the writer.

        This method should be implemented by subclasses.
        """
    raise NotImplementedError('Subclasses of FileBasedDatasource must implement _write_files().')