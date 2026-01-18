import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class BaseCompressedFileFileSystem(AbstractArchiveFileSystem):
    """Read contents of compressed file as a filesystem with one file inside."""
    root_marker = ''
    protocol: str = None
    compression: str = None
    extension: str = None

    def __init__(self, fo: str='', target_protocol: Optional[str]=None, target_options: Optional[dict]=None, **kwargs):
        """
        The compressed file system can be instantiated from any compressed file.
        It reads the contents of compressed file as a filesystem with one file inside, as if it was an archive.

        The single file inside the filesystem is named after the compresssed file,
        without the compression extension at the end of the filename.

        Args:
            fo (:obj:``str``): Path to compressed file. Will fetch file using ``fsspec.open()``
            mode (:obj:``str``): Currently, only 'rb' accepted
            target_protocol(:obj:``str``, optional): To override the FS protocol inferred from a URL.
            target_options (:obj:``dict``, optional): Kwargs passed when instantiating the target FS.
        """
        super().__init__(self, **kwargs)
        self.file = fsspec.open(fo, mode='rb', protocol=target_protocol, compression=self.compression, client_kwargs={'requote_redirect_url': False, 'trust_env': True, **(target_options or {}).pop('client_kwargs', {})}, **target_options or {})
        self.compressed_name = os.path.basename(self.file.path.split('::')[0])
        self.uncompressed_name = self.compressed_name[:self.compressed_name.rindex('.')] if '.' in self.compressed_name else self.compressed_name
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path).lstrip('/')

    def _get_dirs(self):
        if self.dir_cache is None:
            f = {**self.file.fs.info(self.file.path), 'name': self.uncompressed_name}
            self.dir_cache = {f['name']: f}

    def cat(self, path: str):
        return self.file.open().read()

    def _open(self, path: str, mode: str='rb', block_size=None, autocommit=True, cache_options=None, **kwargs):
        path = self._strip_protocol(path)
        if mode != 'rb':
            raise ValueError(f"Tried to read with mode {mode} on file {self.file.path} opened with mode 'rb'")
        return self.file.open()