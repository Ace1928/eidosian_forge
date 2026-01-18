from __future__ import annotations
import copy
import json
from .nbbase import from_dict
from .rwbase import NotebookReader, NotebookWriter, rejoin_lines, restore_bytes, split_lines
class BytesEncoder(json.JSONEncoder):
    """A JSON encoder that accepts b64 (and other *ascii*) bytestrings."""

    def default(self, obj):
        """The default value of an object."""
        if isinstance(obj, bytes):
            return obj.decode('ascii')
        return json.JSONEncoder.default(self, obj)