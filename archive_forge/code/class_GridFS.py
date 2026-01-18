from __future__ import annotations
from collections import abc
from typing import Any, Mapping, Optional, cast
from bson.objectid import ObjectId
from gridfs.errors import NoFile
from gridfs.grid_file import (
from pymongo import ASCENDING, DESCENDING, _csot
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.common import validate_string
from pymongo.database import Database
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import _ServerMode
from pymongo.write_concern import WriteConcern
class GridFS:
    """An instance of GridFS on top of a single Database."""

    def __init__(self, database: Database, collection: str='fs'):
        """Create a new instance of :class:`GridFS`.

        Raises :class:`TypeError` if `database` is not an instance of
        :class:`~pymongo.database.Database`.

        :Parameters:
          - `database`: database to use
          - `collection` (optional): root collection to use

        .. versionchanged:: 4.0
           Removed the `disable_md5` parameter. See
           :ref:`removed-gridfs-checksum` for details.

        .. versionchanged:: 3.11
           Running a GridFS operation in a transaction now always raises an
           error. GridFS does not support multi-document transactions.

        .. versionchanged:: 3.7
           Added the `disable_md5` parameter.

        .. versionchanged:: 3.1
           Indexes are only ensured on the first write to the DB.

        .. versionchanged:: 3.0
           `database` must use an acknowledged
           :attr:`~pymongo.database.Database.write_concern`

        .. seealso:: The MongoDB documentation on `gridfs <https://dochub.mongodb.org/core/gridfs>`_.
        """
        if not isinstance(database, Database):
            raise TypeError('database must be an instance of Database')
        database = _clear_entity_type_registry(database)
        if not database.write_concern.acknowledged:
            raise ConfigurationError('database must use acknowledged write_concern')
        self.__collection = database[collection]
        self.__files = self.__collection.files
        self.__chunks = self.__collection.chunks

    def new_file(self, **kwargs: Any) -> GridIn:
        """Create a new file in GridFS.

        Returns a new :class:`~gridfs.grid_file.GridIn` instance to
        which data can be written. Any keyword arguments will be
        passed through to :meth:`~gridfs.grid_file.GridIn`.

        If the ``"_id"`` of the file is manually specified, it must
        not already exist in GridFS. Otherwise
        :class:`~gridfs.errors.FileExists` is raised.

        :Parameters:
          - `**kwargs` (optional): keyword arguments for file creation
        """
        return GridIn(self.__collection, **kwargs)

    def put(self, data: Any, **kwargs: Any) -> Any:
        """Put data in GridFS as a new file.

        Equivalent to doing::

          with fs.new_file(**kwargs) as f:
              f.write(data)

        `data` can be either an instance of :class:`bytes` or a file-like
        object providing a :meth:`read` method. If an `encoding` keyword
        argument is passed, `data` can also be a :class:`str` instance, which
        will be encoded as `encoding` before being written. Any keyword
        arguments will be passed through to the created file - see
        :meth:`~gridfs.grid_file.GridIn` for possible arguments. Returns the
        ``"_id"`` of the created file.

        If the ``"_id"`` of the file is manually specified, it must
        not already exist in GridFS. Otherwise
        :class:`~gridfs.errors.FileExists` is raised.

        :Parameters:
          - `data`: data to be written as a file.
          - `**kwargs` (optional): keyword arguments for file creation

        .. versionchanged:: 3.0
           w=0 writes to GridFS are now prohibited.
        """
        with GridIn(self.__collection, **kwargs) as grid_file:
            grid_file.write(data)
            return grid_file._id

    def get(self, file_id: Any, session: Optional[ClientSession]=None) -> GridOut:
        """Get a file from GridFS by ``"_id"``.

        Returns an instance of :class:`~gridfs.grid_file.GridOut`,
        which provides a file-like interface for reading.

        :Parameters:
          - `file_id`: ``"_id"`` of the file to get
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
        gout = GridOut(self.__collection, file_id, session=session)
        gout._ensure_file()
        return gout

    def get_version(self, filename: Optional[str]=None, version: Optional[int]=-1, session: Optional[ClientSession]=None, **kwargs: Any) -> GridOut:
        """Get a file from GridFS by ``"filename"`` or metadata fields.

        Returns a version of the file in GridFS whose filename matches
        `filename` and whose metadata fields match the supplied keyword
        arguments, as an instance of :class:`~gridfs.grid_file.GridOut`.

        Version numbering is a convenience atop the GridFS API provided
        by MongoDB. If more than one file matches the query (either by
        `filename` alone, by metadata fields, or by a combination of
        both), then version ``-1`` will be the most recently uploaded
        matching file, ``-2`` the second most recently
        uploaded, etc. Version ``0`` will be the first version
        uploaded, ``1`` the second version, etc. So if three versions
        have been uploaded, then version ``0`` is the same as version
        ``-3``, version ``1`` is the same as version ``-2``, and
        version ``2`` is the same as version ``-1``.

        Raises :class:`~gridfs.errors.NoFile` if no such version of
        that file exists.

        :Parameters:
          - `filename`: ``"filename"`` of the file to get, or `None`
          - `version` (optional): version of the file to get (defaults
            to -1, the most recent version uploaded)
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`
          - `**kwargs` (optional): find files by custom metadata.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.1
           ``get_version`` no longer ensures indexes.
        """
        query = kwargs
        if filename is not None:
            query['filename'] = filename
        _disallow_transactions(session)
        cursor = self.__files.find(query, session=session)
        if version is None:
            version = -1
        if version < 0:
            skip = abs(version) - 1
            cursor.limit(-1).skip(skip).sort('uploadDate', DESCENDING)
        else:
            cursor.limit(-1).skip(version).sort('uploadDate', ASCENDING)
        try:
            doc = next(cursor)
            return GridOut(self.__collection, file_document=doc, session=session)
        except StopIteration:
            raise NoFile('no version %d for filename %r' % (version, filename)) from None

    def get_last_version(self, filename: Optional[str]=None, session: Optional[ClientSession]=None, **kwargs: Any) -> GridOut:
        """Get the most recent version of a file in GridFS by ``"filename"``
        or metadata fields.

        Equivalent to calling :meth:`get_version` with the default
        `version` (``-1``).

        :Parameters:
          - `filename`: ``"filename"`` of the file to get, or `None`
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`
          - `**kwargs` (optional): find files by custom metadata.

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
        return self.get_version(filename=filename, session=session, **kwargs)

    def delete(self, file_id: Any, session: Optional[ClientSession]=None) -> None:
        """Delete a file from GridFS by ``"_id"``.

        Deletes all data belonging to the file with ``"_id"``:
        `file_id`.

        .. warning:: Any processes/threads reading from the file while
           this method is executing will likely see an invalid/corrupt
           file. Care should be taken to avoid concurrent reads to a file
           while it is being deleted.

        .. note:: Deletes of non-existent files are considered successful
           since the end result is the same: no file with that _id remains.

        :Parameters:
          - `file_id`: ``"_id"`` of the file to delete
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.1
           ``delete`` no longer ensures indexes.
        """
        _disallow_transactions(session)
        self.__files.delete_one({'_id': file_id}, session=session)
        self.__chunks.delete_many({'files_id': file_id}, session=session)

    def list(self, session: Optional[ClientSession]=None) -> list[str]:
        """List the names of all files stored in this instance of
        :class:`GridFS`.

        :Parameters:
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.1
           ``list`` no longer ensures indexes.
        """
        _disallow_transactions(session)
        return [name for name in self.__files.distinct('filename', session=session) if name is not None]

    def find_one(self, filter: Optional[Any]=None, session: Optional[ClientSession]=None, *args: Any, **kwargs: Any) -> Optional[GridOut]:
        """Get a single file from gridfs.

        All arguments to :meth:`find` are also valid arguments for
        :meth:`find_one`, although any `limit` argument will be
        ignored. Returns a single :class:`~gridfs.grid_file.GridOut`,
        or ``None`` if no matching file is found. For example::

            file = fs.find_one({"filename": "lisa.txt"})

        :Parameters:
          - `filter` (optional): a dictionary specifying
            the query to be performing OR any other type to be used as
            the value for a query for ``"_id"`` in the file collection.
          - `*args` (optional): any additional positional arguments are
            the same as the arguments to :meth:`find`.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`
          - `**kwargs` (optional): any additional keyword arguments
            are the same as the arguments to :meth:`find`.

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
        if filter is not None and (not isinstance(filter, abc.Mapping)):
            filter = {'_id': filter}
        _disallow_transactions(session)
        for f in self.find(filter, *args, session=session, **kwargs):
            return f
        return None

    def find(self, *args: Any, **kwargs: Any) -> GridOutCursor:
        """Query GridFS for files.

        Returns a cursor that iterates across files matching
        arbitrary queries on the files collection. Can be combined
        with other modifiers for additional control. For example::

          for grid_out in fs.find({"filename": "lisa.txt"},
                                  no_cursor_timeout=True):
              data = grid_out.read()

        would iterate through all versions of "lisa.txt" stored in GridFS.
        Note that setting no_cursor_timeout to True may be important to
        prevent the cursor from timing out during long multi-file processing
        work.

        As another example, the call::

          most_recent_three = fs.find().sort("uploadDate", -1).limit(3)

        would return a cursor to the three most recently uploaded files
        in GridFS.

        Follows a similar interface to
        :meth:`~pymongo.collection.Collection.find`
        in :class:`~pymongo.collection.Collection`.

        If a :class:`~pymongo.client_session.ClientSession` is passed to
        :meth:`find`, all returned :class:`~gridfs.grid_file.GridOut` instances
        are associated with that session.

        :Parameters:
          - `filter` (optional): A query document that selects which files
            to include in the result set. Can be an empty document to include
            all files.
          - `skip` (optional): the number of files to omit (from
            the start of the result set) when returning the results
          - `limit` (optional): the maximum number of results to
            return
          - `no_cursor_timeout` (optional): if False (the default), any
            returned cursor is closed by the server after 10 minutes of
            inactivity. If set to True, the returned cursor will never
            time out on the server. Care should be taken to ensure that
            cursors with no_cursor_timeout turned on are properly closed.
          - `sort` (optional): a list of (key, direction) pairs
            specifying the sort order for this query. See
            :meth:`~pymongo.cursor.Cursor.sort` for details.

        Raises :class:`TypeError` if any of the arguments are of
        improper type. Returns an instance of
        :class:`~gridfs.grid_file.GridOutCursor`
        corresponding to this query.

        .. versionchanged:: 3.0
           Removed the read_preference, tag_sets, and
           secondary_acceptable_latency_ms options.
        .. versionadded:: 2.7
        .. seealso:: The MongoDB documentation on `find <https://dochub.mongodb.org/core/find>`_.
        """
        return GridOutCursor(self.__collection, *args, **kwargs)

    def exists(self, document_or_id: Optional[Any]=None, session: Optional[ClientSession]=None, **kwargs: Any) -> bool:
        """Check if a file exists in this instance of :class:`GridFS`.

        The file to check for can be specified by the value of its
        ``_id`` key, or by passing in a query document. A query
        document can be passed in as dictionary, or by using keyword
        arguments. Thus, the following three calls are equivalent:

        >>> fs.exists(file_id)
        >>> fs.exists({"_id": file_id})
        >>> fs.exists(_id=file_id)

        As are the following two calls:

        >>> fs.exists({"filename": "mike.txt"})
        >>> fs.exists(filename="mike.txt")

        And the following two:

        >>> fs.exists({"foo": {"$gt": 12}})
        >>> fs.exists(foo={"$gt": 12})

        Returns ``True`` if a matching file exists, ``False``
        otherwise. Calls to :meth:`exists` will not automatically
        create appropriate indexes; application developers should be
        sure to create indexes if needed and as appropriate.

        :Parameters:
          - `document_or_id` (optional): query document, or _id of the
            document to check for
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`
          - `**kwargs` (optional): keyword arguments are used as a
            query document, if they're present.

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
        _disallow_transactions(session)
        if kwargs:
            f = self.__files.find_one(kwargs, ['_id'], session=session)
        else:
            f = self.__files.find_one(document_or_id, ['_id'], session=session)
        return f is not None