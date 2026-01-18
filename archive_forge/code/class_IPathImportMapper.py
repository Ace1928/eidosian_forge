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
class IPathImportMapper(Interface):
    """
    This is an internal interface, used to map importers to factories for
    FilePath-like objects.
    """

    def mapPath(pathLikeString):
        """
        Return a FilePath-like object.

        @param pathLikeString: a path-like string, like one that might be
        passed to an import hook.

        @return: a L{FilePath}, or something like it (currently only a
        L{ZipPath}, but more might be added later).
        """