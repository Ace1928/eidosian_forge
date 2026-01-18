import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def _index_file(self, name, location):
    if name not in self._index:
        self._index[name] = location