import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
def _get_fileinfo_sizes(self, pstat, subinfo, packinfo, folder, packsizes, unpacksizes, file_in_solid, numinstreams):
    if pstat.input == 0:
        folder.solid = subinfo.num_unpackstreams_folders[pstat.folder] > 1
    maxsize = folder.solid and packinfo.packsizes[pstat.stream] or None
    uncompressed = unpacksizes[pstat.outstreams]
    if file_in_solid > 0:
        compressed = None
    elif pstat.stream < len(packsizes):
        compressed = packsizes[pstat.stream]
    else:
        compressed = uncompressed
    packsize = packsizes[pstat.stream:pstat.stream + numinstreams]
    return (maxsize, compressed, uncompressed, packsize, folder.solid)