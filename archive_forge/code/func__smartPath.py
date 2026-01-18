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
def _smartPath(self, pathName):
    """
        Given a path entry from sys.path which may refer to an importer,
        return the appropriate FilePath-like instance.

        @param pathName: a str describing the path.

        @return: a FilePath-like object.
        """
    importr = self.importerCache.get(pathName, _nothing)
    if importr is _nothing:
        for hook in self.sysPathHooks:
            try:
                importr = hook(pathName)
            except ImportError:
                pass
        if importr is _nothing:
            importr = None
    return IPathImportMapper(importr, _theDefaultMapper).mapPath(pathName)