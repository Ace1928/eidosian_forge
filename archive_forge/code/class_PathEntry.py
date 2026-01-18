from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
class PathEntry(_ModuleIteratorHelper):
    """
    I am a proxy for a single entry on sys.path.

    @ivar filePath: a FilePath-like object pointing at the filesystem location
    or archive file where this path entry is stored.

    @ivar pythonPath: a PythonPath instance.
    """

    def __init__(self, filePath, pythonPath):
        """
        Create a PathEntry.  This is a private constructor.
        """
        self.filePath = filePath
        self.pythonPath = pythonPath

    def _getEntry(self):
        return self

    def __repr__(self) -> str:
        return f'PathEntry<{self.filePath!r}>'

    def _packagePaths(self):
        yield self.filePath