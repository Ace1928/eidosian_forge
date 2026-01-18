from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
def _chunk_read(response, local_file, chunk_size=8192, initial_size=0):
    """Download a file chunk by chunk and show advancement

    Can also be used when resuming downloads over http.

    Parameters
    ----------
    response: urllib.response.addinfourl
        Response to the download request in order to get file size.
    local_file: file
        Hard disk file where data should be written.
    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192
    initial_size: int, optional
        If resuming, indicate the initial size of the file.
    """
    bytes_so_far = initial_size
    total_size = int(response.headers['Content-Length'].strip())
    total_size += initial_size
    progress = StdoutProgressIndicator('Downloading')
    progress.start('', 'bytes', total_size)
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        if not chunk:
            break
        _chunk_write(chunk, local_file, progress)
    progress.finish('Done')