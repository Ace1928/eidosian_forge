import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class ZstdFileSystem(BaseCompressedFileFileSystem):
    """
    Read contents of zstd file as a filesystem with one file inside.

    Note that reading in binary mode with fsspec isn't supported yet:
    https://github.com/indygreg/python-zstandard/issues/136
    """
    protocol = 'zstd'
    compression = 'zstd'
    extension = '.zst'

    def __init__(self, fo: str, mode: str='rb', target_protocol: Optional[str]=None, target_options: Optional[dict]=None, block_size: int=DEFAULT_BLOCK_SIZE, **kwargs):
        super().__init__(fo=fo, mode=mode, target_protocol=target_protocol, target_options=target_options, block_size=block_size, **kwargs)
        _enter = self.file.__enter__

        class WrappedFile:

            def __init__(self, file_):
                self._file = file_

            def __enter__(self):
                self._file.__enter__()
                return self

            def __exit__(self, *args, **kwargs):
                self._file.__exit__(*args, **kwargs)

            def __iter__(self):
                return iter(self._file)

            def __next__(self):
                return next(self._file)

            def __getattr__(self, attr):
                return getattr(self._file, attr)

        def fixed_enter(*args, **kwargs):
            return WrappedFile(_enter(*args, **kwargs))
        self.file.__enter__ = fixed_enter