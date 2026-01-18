import asyncio
from functools import partial, singledispatch
from io import BufferedRandom, BufferedReader, BufferedWriter, FileIO, TextIOBase
from tempfile import NamedTemporaryFile as syncNamedTemporaryFile
from tempfile import SpooledTemporaryFile as syncSpooledTemporaryFile
from tempfile import TemporaryDirectory as syncTemporaryDirectory
from tempfile import TemporaryFile as syncTemporaryFile
from tempfile import _TemporaryFileWrapper as syncTemporaryFileWrapper
from ..base import AiofilesContextManager
from ..threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from ..threadpool.text import AsyncTextIOWrapper
from .temptypes import AsyncSpooledTemporaryFile, AsyncTemporaryDirectory
import sys
def SpooledTemporaryFile(max_size=0, mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, loop=None, executor=None):
    """Async open a spooled temporary file"""
    return AiofilesContextManager(_spooled_temporary_file(max_size=max_size, mode=mode, buffering=buffering, encoding=encoding, newline=newline, suffix=suffix, prefix=prefix, dir=dir, loop=loop, executor=executor))