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
def _defaultSysPathFactory():
    """
    Provide the default behavior of PythonPath's sys.path factory, which is to
    return the current value of sys.path.

    @return: L{sys.path}
    """
    return sys.path