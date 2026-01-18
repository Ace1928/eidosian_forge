import base64
import urllib
import requests
import requests.exceptions
from requests.adapters import HTTPAdapter, Retry
from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
class DatabricksFile(AbstractBufferedFile):
    """
    Helper class for files referenced in the DatabricksFileSystem.
    """
    DEFAULT_BLOCK_SIZE = 1 * 2 ** 20

    def __init__(self, fs, path, mode='rb', block_size='default', autocommit=True, cache_type='readahead', cache_options=None, **kwargs):
        """
        Create a new instance of the DatabricksFile.

        The blocksize needs to be the default one.
        """
        if block_size is None or block_size == 'default':
            block_size = self.DEFAULT_BLOCK_SIZE
        assert block_size == self.DEFAULT_BLOCK_SIZE, f'Only the default block size is allowed, not {block_size}'
        super().__init__(fs, path, mode=mode, block_size=block_size, autocommit=autocommit, cache_type=cache_type, cache_options=cache_options or {}, **kwargs)

    def _initiate_upload(self):
        """Internal function to start a file upload"""
        self.handle = self.fs._create_handle(self.path)

    def _upload_chunk(self, final=False):
        """Internal function to add a chunk of data to a started upload"""
        self.buffer.seek(0)
        data = self.buffer.getvalue()
        data_chunks = [data[start:end] for start, end in self._to_sized_blocks(len(data))]
        for data_chunk in data_chunks:
            self.fs._add_data(handle=self.handle, data=data_chunk)
        if final:
            self.fs._close_handle(handle=self.handle)
            return True

    def _fetch_range(self, start, end):
        """Internal function to download a block of data"""
        return_buffer = b''
        length = end - start
        for chunk_start, chunk_end in self._to_sized_blocks(length, start):
            return_buffer += self.fs._get_data(path=self.path, start=chunk_start, end=chunk_end)
        return return_buffer

    def _to_sized_blocks(self, length, start=0):
        """Helper function to split a range from 0 to total_length into bloksizes"""
        end = start + length
        for data_chunk in range(start, end, self.blocksize):
            data_start = data_chunk
            data_end = min(end, data_chunk + self.blocksize)
            yield (data_start, data_end)