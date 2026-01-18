from __future__ import annotations
from base64 import decodebytes, encodebytes
class NotebookWriter:
    """A class for writing notebooks."""

    def writes(self, nb, **kwargs):
        """Write a notebook to a string."""
        msg = 'loads must be implemented in a subclass'
        raise NotImplementedError(msg)

    def write(self, nb, fp, **kwargs):
        """Write a notebook to a file like object"""
        return fp.write(self.writes(nb, **kwargs))