import fnmatch
import io
import os
import shutil
import tarfile
from typing import Optional, Tuple, Dict, Generator, Union, List
import ray
from ray.util.annotations import DeveloperAPI
from ray.air._internal.filelock import TempFileLock
from ray.air.util.node import _get_node_id_from_node_ip, _force_on_node
@ray.remote
class _PackActor:
    """Actor wrapping around a packing job.

    This actor is used for chunking the packed data into smaller chunks that
    can be transferred via the object store more efficiently.

    The actor will start packing the directory when initialized, and separate
    chunks can be received by calling the remote ``next()`` task.

    Args:
        source_dir: Path to local directory to pack into tarfile.
        exclude: Pattern of files to exclude, e.g.
            ``["*/checkpoint_*]`` to exclude trial checkpoints.
        files_stats: Dict of relative filenames mapping to a tuple of
            (mtime, filesize). Only files that differ from these stats
            will be packed.
        chunk_size_bytes: Cut bytes stream into chunks of this size in bytes.
        max_size_bytes: If packed data exceeds this value, raise an error before
            transfer. If ``None``, no limit is enforced.
    """

    def __init__(self, source_dir: str, exclude: Optional[List]=None, files_stats: Optional[Dict[str, Tuple[float, int]]]=None, chunk_size_bytes: int=_DEFAULT_CHUNK_SIZE_BYTES, max_size_bytes: Optional[int]=_DEFAULT_MAX_SIZE_BYTES):
        self.stream = _pack_dir(source_dir=source_dir, exclude=exclude, files_stats=files_stats)
        self.stream.seek(0, 2)
        file_size = self.stream.tell()
        if max_size_bytes and file_size > max_size_bytes:
            raise RuntimeError(f'Packed directory {source_dir} content has a size of {_gib_string(file_size)}, which exceeds the limit of {_gib_string(max_size_bytes)}. Please check the directory contents. If you want to transfer everything, you can increase or disable the limit by passing the `max_size` argument.')
        self.chunk_size = chunk_size_bytes
        self.max_size = max_size_bytes
        self.iter = None

    def get_full_data(self) -> bytes:
        return self.stream.getvalue()

    def _chunk_generator(self) -> Generator[bytes, None, None]:
        self.stream.seek(0)
        data = self.stream.read(self.chunk_size)
        while data:
            yield data
            data = self.stream.read(self.chunk_size)

    def next(self) -> Optional[bytes]:
        if not self.iter:
            self.iter = iter(self._chunk_generator())
        try:
            return next(self.iter)
        except StopIteration:
            return None