import os
import zstandard
import ujson as json
import time
import tarfile
import codecs
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path
def _stream_data_threaded(self, q, get_meta=False):
    for data in self._stream_data(get_meta):
        q.put(data)
    q.put(None)