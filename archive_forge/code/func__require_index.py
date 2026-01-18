import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def _require_index(self):
    if self._index is None:
        self.reindex()