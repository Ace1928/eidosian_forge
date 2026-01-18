from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class InvalidChunkSizeError(Error):
    """The given chunksize is not valid."""
    pass