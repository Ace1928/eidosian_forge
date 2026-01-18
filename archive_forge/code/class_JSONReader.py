from __future__ import annotations
import copy
import json
from .nbbase import from_dict
from .rwbase import NotebookReader, NotebookWriter, rejoin_lines, restore_bytes, split_lines
class JSONReader(NotebookReader):
    """A JSON notebook reader."""

    def reads(self, s, **kwargs):
        """Convert a string to a notebook."""
        nb = json.loads(s, **kwargs)
        nb = self.to_notebook(nb, **kwargs)
        return nb

    def to_notebook(self, d, **kwargs):
        """Convert a string to a notebook."""
        return restore_bytes(rejoin_lines(from_dict(d)))