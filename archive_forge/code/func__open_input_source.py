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
def _open_input_source(self, filesystem: 'pyarrow.fs.FileSystem', path: str, **open_args) -> 'pyarrow.NativeFile':
    """Opens a source path for reading and returns the associated Arrow NativeFile.

        The default implementation opens the source path as a sequential input stream,
        using ctx.streaming_read_buffer_size as the buffer size if none is given by the
        caller.

        Implementations that do not support streaming reads (e.g. that require random
        access) should override this method.
        """
    import pyarrow as pa
    from pyarrow.fs import HadoopFileSystem
    compression = open_args.get('compression', None)
    if compression is None:
        try:
            compression = pa.Codec.detect(path).name
        except (ValueError, TypeError):
            import pathlib
            suffix = pathlib.Path(path).suffix
            if suffix and suffix[1:] == 'snappy':
                compression = 'snappy'
            else:
                compression = None
    buffer_size = open_args.pop('buffer_size', None)
    if buffer_size is None:
        ctx = DataContext.get_current()
        buffer_size = ctx.streaming_read_buffer_size
    if compression == 'snappy':
        open_args['compression'] = None
    else:
        open_args['compression'] = compression
    file = filesystem.open_input_stream(path, buffer_size=buffer_size, **open_args)
    if compression == 'snappy':
        import snappy
        stream = io.BytesIO()
        if isinstance(filesystem, HadoopFileSystem):
            snappy.hadoop_snappy.stream_decompress(src=file, dst=stream)
        else:
            snappy.stream_decompress(src=file, dst=stream)
        stream.seek(0)
        file = pa.PythonFile(stream, mode='r')
    return file