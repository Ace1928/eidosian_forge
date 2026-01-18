import base64
import urllib
import requests
import requests.exceptions
from requests.adapters import HTTPAdapter, Retry
from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
def _to_sized_blocks(self, length, start=0):
    """Helper function to split a range from 0 to total_length into bloksizes"""
    end = start + length
    for data_chunk in range(start, end, self.blocksize):
        data_start = data_chunk
        data_end = min(end, data_chunk + self.blocksize)
        yield (data_start, data_end)