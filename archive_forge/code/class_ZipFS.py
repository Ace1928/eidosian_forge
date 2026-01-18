from __future__ import print_function, unicode_literals
import sys
import typing
import six
import zipfile
from datetime import datetime
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_zip
from .enums import ResourceType, Seek
from .info import Info
from .iotools import RawWrapper
from .memoryfs import MemoryFS
from .opener import open_fs
from .path import dirname, forcedir, normpath, relpath
from .permissions import Permissions
from .time import datetime_to_epoch
from .wrapfs import WrapFS
class ZipFS(WrapFS):
    """Read and write zip files.

    There are two ways to open a `ZipFS` for the use cases of reading
    a zip file, and creating a new one.

    If you open the `ZipFS` with  ``write`` set to `False` (the default)
    then the filesystem will be a read-only filesystem which maps to
    the files and directories within the zip file. Files are
    decompressed on the fly when you open them.

    Here's how you might extract and print a readme from a zip file::

        with ZipFS('foo.zip') as zip_fs:
            readme = zip_fs.readtext('readme.txt')

    If you open the `ZipFS` with ``write`` set to `True`, then the `ZipFS`
    will be an empty temporary filesystem. Any files / directories you
    create in the `ZipFS` will be written in to a zip file when the `ZipFS`
    is closed.

    Here's how you might write a new zip file containing a ``readme.txt``
    file::

        with ZipFS('foo.zip', write=True) as new_zip:
            new_zip.writetext(
                'readme.txt',
                'This zip file was written by PyFilesystem'
            )


    Arguments:
        file (str or io.IOBase): An OS filename, or an open file object.
        write (bool): Set to `True` to write a new zip file, or `False`
            (default) to read an existing zip file.
        compression (int): Compression to use (one of the constants
            defined in the `zipfile` module in the stdlib).
        temp_fs (str or FS): An FS URL or an FS instance to use to
            store data prior to zipping. Defaults to creating a new
            `~fs.tempfs.TempFS`.

    """

    def __new__(cls, file, write=False, compression=zipfile.ZIP_DEFLATED, encoding='utf-8', temp_fs='temp://__ziptemp__'):
        if write:
            return WriteZipFS(file, compression=compression, encoding=encoding, temp_fs=temp_fs)
        else:
            return ReadZipFS(file, encoding=encoding)
    if typing.TYPE_CHECKING:

        def __init__(self, file, write=False, compression=zipfile.ZIP_DEFLATED, encoding='utf-8', temp_fs='temp://__ziptemp__'):
            pass