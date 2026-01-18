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
def _after_write(self, insize, foutsize, crc):
    self.header.main_streams.substreamsinfo.digestsdefined.append(True)
    self.header.main_streams.substreamsinfo.digests.append(crc)
    if self.header.main_streams.substreamsinfo.unpacksizes is None:
        self.header.main_streams.substreamsinfo.unpacksizes = [insize]
    else:
        self.header.main_streams.substreamsinfo.unpacksizes.append(insize)
    if self.header.main_streams.substreamsinfo.num_unpackstreams_folders is None:
        self.header.main_streams.substreamsinfo.num_unpackstreams_folders = [1]
    else:
        self.header.main_streams.substreamsinfo.num_unpackstreams_folders[-1] += 1
    return (foutsize, crc)