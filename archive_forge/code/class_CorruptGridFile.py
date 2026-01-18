from __future__ import annotations
from pymongo.errors import PyMongoError
class CorruptGridFile(GridFSError):
    """Raised when a file in :class:`~gridfs.GridFS` is malformed."""