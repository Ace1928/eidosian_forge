import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
class CustomFS(FileSystem):

    def __init__(self):
        self.path = None
        self.mode = None

    def open(self, path, mode='rb'):
        self.path = path
        self.mode = mode
        return out