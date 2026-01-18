from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
class TarFS(WrapFS):
    """Read and write tar files.

    There are two ways to open a `TarFS` for the use cases of reading
    a tar file, and creating a new one.

    If you open the `TarFS` with  ``write`` set to `False` (the
    default), then the filesystem will be a read only filesystem which
    maps to the files and directories within the tar file. Files are
    decompressed on the fly when you open them.

    Here's how you might extract and print a readme from a tar file::

        with TarFS('foo.tar.gz') as tar_fs:
            readme = tar_fs.readtext('readme.txt')

    If you open the TarFS with ``write`` set to `True`, then the `TarFS`
    will be a empty temporary filesystem. Any files / directories you
    create in the `TarFS` will be written in to a tar file when the `TarFS`
    is closed. The compression is set from the new file name but may be
    set manually with the ``compression`` argument.

    Here's how you might write a new tar file containing a readme.txt
    file::

        with TarFS('foo.tar.xz', write=True) as new_tar:
            new_tar.writetext(
                'readme.txt',
                'This tar file was written by PyFilesystem'
            )

    Arguments:
        file (str or io.IOBase): An OS filename, or an open file handle.
        write (bool): Set to `True` to write a new tar file, or
            use default (`False`) to read an existing tar file.
        compression (str, optional): Compression to use (one of the formats
            supported by `tarfile`: ``xz``, ``gz``, ``bz2``, or `None`).
        temp_fs (str): An FS URL or an FS instance to use to store
            data prior to tarring. Defaults to creating a new
            `~fs.tempfs.TempFS`.

    """
    _compression_formats = {'xz': ('.tar.xz', '.txz'), 'bz2': ('.tar.bz2', '.tbz'), 'gz': ('.tar.gz', '.tgz')}

    def __new__(cls, file, write=False, compression=None, encoding='utf-8', temp_fs='temp://__tartemp__'):
        if isinstance(file, (six.text_type, six.binary_type)):
            file = os.path.expanduser(file)
            filename = file
        else:
            filename = getattr(file, 'name', '')
        if write and compression is None:
            compression = None
            for comp, extensions in six.iteritems(cls._compression_formats):
                if filename.endswith(extensions):
                    compression = comp
                    break
        if write:
            return WriteTarFS(file, compression=compression, encoding=encoding, temp_fs=temp_fs)
        else:
            return ReadTarFS(file, encoding=encoding)
    if typing.TYPE_CHECKING:

        def __init__(self, file, write=False, compression=None, encoding='utf-8', temp_fs='temp://__tartemp__'):
            pass