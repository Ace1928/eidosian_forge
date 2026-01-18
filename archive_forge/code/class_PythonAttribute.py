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
class PythonAttribute:
    """
    I represent a function, class, or other object that is present.

    @ivar name: the fully-qualified python name of this attribute.

    @ivar onObject: a reference to a PythonModule or other PythonAttribute that
    is this attribute's logical parent.

    @ivar name: the fully qualified python name of the attribute represented by
    this class.
    """

    def __init__(self, name: str, onObject: PythonAttribute, loaded: bool, pythonValue: object) -> None:
        """
        Create a PythonAttribute.  This is a private constructor.  Do not construct
        me directly, use PythonModule.iterAttributes.

        @param name: the FQPN
        @param onObject: see ivar
        @param loaded: always True, for now
        @param pythonValue: the value of the attribute we're pointing to.
        """
        self.name: str = name
        self.onObject = onObject
        self._loaded = loaded
        self.pythonValue = pythonValue

    def __repr__(self) -> str:
        return f'PythonAttribute<{self.name!r}>'

    def isLoaded(self):
        """
        Return a boolean describing whether the attribute this describes has
        actually been loaded into memory by importing its module.

        Note: this currently always returns true; there is no Python parser
        support in this module yet.
        """
        return self._loaded

    def load(self, default=_nothing):
        """
        Load the value associated with this attribute.

        @return: an arbitrary Python object, or 'default' if there is an error
        loading it.
        """
        return self.pythonValue

    def iterAttributes(self):
        for name, val in inspect.getmembers(self.load()):
            yield PythonAttribute(self.name + '.' + name, self, True, val)