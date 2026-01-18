from __future__ import annotations
from pymongo.errors import PyMongoError
class GridFSError(PyMongoError):
    """Base class for all GridFS exceptions."""