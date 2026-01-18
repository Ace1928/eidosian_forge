from __future__ import annotations
import datetime
import io
import math
import os
from typing import Any, Iterable, Mapping, NoReturn, Optional
from bson.binary import Binary
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.son import SON
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import (
from pymongo.read_preferences import ReadPreference
class GridOut(io.IOBase):
    """Class to read data out of GridFS."""

    def __init__(self, root_collection: Collection, file_id: Optional[int]=None, file_document: Optional[Any]=None, session: Optional[ClientSession]=None) -> None:
        """Read a file from GridFS

        Application developers should generally not need to
        instantiate this class directly - instead see the methods
        provided by :class:`~gridfs.GridFS`.

        Either `file_id` or `file_document` must be specified,
        `file_document` will be given priority if present. Raises
        :class:`TypeError` if `root_collection` is not an instance of
        :class:`~pymongo.collection.Collection`.

        :Parameters:
          - `root_collection`: root collection to read from
          - `file_id` (optional): value of ``"_id"`` for the file to read
          - `file_document` (optional): file document from
            `root_collection.files`
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession` to use for all
            commands

        .. versionchanged:: 3.8
           For better performance and to better follow the GridFS spec,
           :class:`GridOut` now uses a single cursor to read all the chunks in
           the file.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.0
           Creating a GridOut does not immediately retrieve the file metadata
           from the server. Metadata is fetched when first needed.
        """
        if not isinstance(root_collection, Collection):
            raise TypeError('root_collection must be an instance of Collection')
        _disallow_transactions(session)
        root_collection = _clear_entity_type_registry(root_collection)
        super().__init__()
        self.__chunks = root_collection.chunks
        self.__files = root_collection.files
        self.__file_id = file_id
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        self.__chunk_iter = None
        self.__position = 0
        self._file = file_document
        self._session = session
    _id: Any = _grid_out_property('_id', "The ``'_id'`` value for this file.")
    filename: str = _grid_out_property('filename', 'Name of this file.')
    name: str = _grid_out_property('filename', 'Alias for `filename`.')
    content_type: Optional[str] = _grid_out_property('contentType', 'DEPRECATED, will be removed in PyMongo 5.0. Mime-type for this file.')
    length: int = _grid_out_property('length', 'Length (in bytes) of this file.')
    chunk_size: int = _grid_out_property('chunkSize', 'Chunk size for this file.')
    upload_date: datetime.datetime = _grid_out_property('uploadDate', 'Date that this file was first uploaded.')
    aliases: Optional[list[str]] = _grid_out_property('aliases', 'DEPRECATED, will be removed in PyMongo 5.0. List of aliases for this file.')
    metadata: Optional[Mapping[str, Any]] = _grid_out_property('metadata', 'Metadata attached to this file.')
    md5: Optional[str] = _grid_out_property('md5', 'DEPRECATED, will be removed in PyMongo 5.0. MD5 of the contents of this file if an md5 sum was created.')
    _file: Any
    __chunk_iter: Any

    def _ensure_file(self) -> None:
        if not self._file:
            _disallow_transactions(self._session)
            self._file = self.__files.find_one({'_id': self.__file_id}, session=self._session)
            if not self._file:
                raise NoFile(f'no file in gridfs collection {self.__files!r} with _id {self.__file_id!r}')

    def __getattr__(self, name: str) -> Any:
        self._ensure_file()
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridOut object has no attribute '%s'" % name)

    def readable(self) -> bool:
        return True

    def readchunk(self) -> bytes:
        """Reads a chunk at a time. If the current position is within a
        chunk the remainder of the chunk is returned.
        """
        received = len(self.__buffer) - self.__buffer_pos
        chunk_data = EMPTY
        chunk_size = int(self.chunk_size)
        if received > 0:
            chunk_data = self.__buffer[self.__buffer_pos:]
        elif self.__position < int(self.length):
            chunk_number = int((received + self.__position) / chunk_size)
            if self.__chunk_iter is None:
                self.__chunk_iter = _GridOutChunkIterator(self, self.__chunks, self._session, chunk_number)
            chunk = self.__chunk_iter.next()
            chunk_data = chunk['data'][self.__position % chunk_size:]
            if not chunk_data:
                raise CorruptGridFile('truncated chunk')
        self.__position += len(chunk_data)
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        return chunk_data

    def _read_size_or_line(self, size: int=-1, line: bool=False) -> bytes:
        """Internal read() and readline() helper."""
        self._ensure_file()
        remainder = int(self.length) - self.__position
        if size < 0 or size > remainder:
            size = remainder
        if size == 0:
            return EMPTY
        received = 0
        data = []
        while received < size:
            needed = size - received
            if self.__buffer:
                buf = self.__buffer
                chunk_start = self.__buffer_pos
                chunk_data = memoryview(buf)[self.__buffer_pos:]
                self.__buffer = EMPTY
                self.__buffer_pos = 0
                self.__position += len(chunk_data)
            else:
                buf = self.readchunk()
                chunk_start = 0
                chunk_data = memoryview(buf)
            if line:
                pos = buf.find(NEWLN, chunk_start, chunk_start + needed) - chunk_start
                if pos >= 0:
                    size = received + pos + 1
                    needed = pos + 1
            if len(chunk_data) > needed:
                data.append(chunk_data[:needed])
                self.__buffer = buf
                self.__buffer_pos = chunk_start + needed
                self.__position -= len(self.__buffer) - self.__buffer_pos
            else:
                data.append(chunk_data)
            received += len(chunk_data)
        if size == remainder and self.__chunk_iter:
            try:
                self.__chunk_iter.next()
            except StopIteration:
                pass
        return b''.join(data)

    def read(self, size: int=-1) -> bytes:
        """Read at most `size` bytes from the file (less if there
        isn't enough data).

        The bytes are returned as an instance of :class:`bytes`
        If `size` is negative or omitted all data is read.

        :Parameters:
          - `size` (optional): the number of bytes to read

        .. versionchanged:: 3.8
           This method now only checks for extra chunks after reading the
           entire file. Previously, this method would check for extra chunks
           on every call.
        """
        return self._read_size_or_line(size=size)

    def readline(self, size: int=-1) -> bytes:
        """Read one line or up to `size` bytes from the file.

        :Parameters:
         - `size` (optional): the maximum number of bytes to read
        """
        return self._read_size_or_line(size=size, line=True)

    def tell(self) -> int:
        """Return the current position of this file."""
        return self.__position

    def seek(self, pos: int, whence: int=_SEEK_SET) -> int:
        """Set the current position of this file.

        :Parameters:
         - `pos`: the position (or offset if using relative
           positioning) to seek to
         - `whence` (optional): where to seek
           from. :attr:`os.SEEK_SET` (``0``) for absolute file
           positioning, :attr:`os.SEEK_CUR` (``1``) to seek relative
           to the current position, :attr:`os.SEEK_END` (``2``) to
           seek relative to the file's end.

        .. versionchanged:: 4.1
           The method now returns the new position in the file, to
           conform to the behavior of :meth:`io.IOBase.seek`.
        """
        if whence == _SEEK_SET:
            new_pos = pos
        elif whence == _SEEK_CUR:
            new_pos = self.__position + pos
        elif whence == _SEEK_END:
            new_pos = int(self.length) + pos
        else:
            raise OSError(22, 'Invalid value for `whence`')
        if new_pos < 0:
            raise OSError(22, 'Invalid value for `pos` - must be positive')
        if new_pos == self.__position:
            return new_pos
        self.__position = new_pos
        self.__buffer = EMPTY
        self.__buffer_pos = 0
        if self.__chunk_iter:
            self.__chunk_iter.close()
            self.__chunk_iter = None
        return new_pos

    def seekable(self) -> bool:
        return True

    def __iter__(self) -> GridOut:
        """Return an iterator over all of this file's data.

        The iterator will return lines (delimited by ``b'\\n'``) of
        :class:`bytes`. This can be useful when serving files
        using a webserver that handles such an iterator efficiently.

        .. versionchanged:: 3.8
           The iterator now raises :class:`CorruptGridFile` when encountering
           any truncated, missing, or extra chunk in a file. The previous
           behavior was to only raise :class:`CorruptGridFile` on a missing
           chunk.

        .. versionchanged:: 4.0
           The iterator now iterates over *lines* in the file, instead
           of chunks, to conform to the base class :py:class:`io.IOBase`.
           Use :meth:`GridOut.readchunk` to read chunk by chunk instead
           of line by line.
        """
        return self

    def close(self) -> None:
        """Make GridOut more generically file-like."""
        if self.__chunk_iter:
            self.__chunk_iter.close()
            self.__chunk_iter = None
        super().close()

    def write(self, value: Any) -> NoReturn:
        raise io.UnsupportedOperation('write')

    def writelines(self, lines: Any) -> NoReturn:
        raise io.UnsupportedOperation('writelines')

    def writable(self) -> bool:
        return False

    def __enter__(self) -> GridOut:
        """Makes it possible to use :class:`GridOut` files
        with the context manager protocol.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Makes it possible to use :class:`GridOut` files
        with the context manager protocol.
        """
        self.close()
        return False

    def fileno(self) -> NoReturn:
        raise io.UnsupportedOperation('fileno')

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def truncate(self, size: Optional[int]=None) -> NoReturn:
        raise io.UnsupportedOperation('truncate')

    def __del__(self) -> None:
        pass