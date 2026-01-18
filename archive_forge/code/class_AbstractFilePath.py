from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
class AbstractFilePath(Generic[AnyStr]):
    """
    Abstract implementation of an L{IFilePath}; must be completed by a
    subclass.

    This class primarily exists to provide common implementations of certain
    methods in L{IFilePath}. It is *not* a required parent class for
    L{IFilePath} implementations, just a useful starting point.

    @ivar path: Subclasses must set this variable.
    """
    Selfish = TypeVar('Selfish', bound='AbstractFilePath[AnyStr]')
    path: AnyStr

    def getAccessTime(self) -> float:
        """
        Subclasses must implement this.

        @see: L{FilePath.getAccessTime}
        """
        raise NotImplementedError()

    def getModificationTime(self) -> float:
        """
        Subclasses must implement this.

        @see: L{FilePath.getModificationTime}
        """
        raise NotImplementedError()

    def getStatusChangeTime(self) -> float:
        """
        Subclasses must implement this.

        @see: L{FilePath.getStatusChangeTime}
        """
        raise NotImplementedError()

    def open(self, mode: FileMode='r') -> IO[bytes]:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def isdir(self) -> bool:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def basename(self) -> AnyStr:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def parent(self) -> AbstractFilePath[AnyStr]:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def listdir(self) -> List[AnyStr]:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def child(self, path: OtherAnyStr) -> AbstractFilePath[OtherAnyStr]:
        """
        Subclasses must implement this.
        """
        raise NotImplementedError()

    def getContent(self) -> bytes:
        """
        Retrieve the contents of the file at this path.

        @return: the contents of the file
        @rtype: L{bytes}
        """
        with self.open() as fp:
            return fp.read()

    def parents(self) -> Iterable[AbstractFilePath[AnyStr]]:
        """
        Retrieve an iterator of all the ancestors of this path.

        @return: an iterator of all the ancestors of this path, from the most
        recent (its immediate parent) to the root of its filesystem.
        """
        path = self
        parent = path.parent()
        while path != parent:
            yield parent
            path = parent
            parent = parent.parent()

    def children(self: _Self) -> Iterable[_Self]:
        """
        List the children of this path object.

        @raise OSError: If an error occurs while listing the directory.  If the
        error is 'serious', meaning that the operation failed due to an access
        violation, exhaustion of some kind of resource (file descriptors or
        memory), OSError or a platform-specific variant will be raised.

        @raise UnlistableError: If the inability to list the directory is due
        to this path not existing or not being a directory, the more specific
        OSError subclass L{UnlistableError} is raised instead.

        @return: an iterable of all currently-existing children of this object.
        """
        try:
            subnames: List[AnyStr] = self.listdir()
        except OSError as ose:
            if getattr(ose, 'winerror', None) in (ERROR_PATH_NOT_FOUND, ERROR_FILE_NOT_FOUND, ERROR_INVALID_NAME, ERROR_DIRECTORY):
                raise UnlistableError(ose)
            if ose.errno in (errno.ENOENT, errno.ENOTDIR):
                raise UnlistableError(ose)
            raise
        result = []
        for name in subnames:
            child: _Self = self.child(name)
            result.append(child)
        return result

    def walk(self: _Self, descend: Optional[Callable[[_Self], bool]]=None) -> Iterable[_Self]:
        """
        Yield myself, then each of my children, and each of those children's
        children in turn.

        The optional argument C{descend} is a predicate that takes a FilePath,
        and determines whether or not that FilePath is traversed/descended
        into.  It will be called with each path for which C{isdir} returns
        C{True}.  If C{descend} is not specified, all directories will be
        traversed (including symbolic links which refer to directories).

        @param descend: A one-argument callable that will return True for
            FilePaths that should be traversed, False otherwise.

        @return: a generator yielding FilePath-like objects.
        """
        yield self
        if self.isdir():
            for c in self.children():
                if descend is None or descend(c):
                    for subc in c.walk(descend):
                        if os.path.realpath(self.path).startswith(os.path.realpath(subc.path)):
                            raise LinkError('Cycle in file graph.')
                        yield subc
                else:
                    yield c

    def sibling(self: _Self, path: OtherAnyStr) -> AbstractFilePath[OtherAnyStr]:
        """
        Return a L{FilePath} with the same directory as this instance but with
        a basename of C{path}.

        @note: for type-checking, subclasses should override this signature to
            make it clear that it returns the subclass and not
            L{AbstractFilePath}.

        @param path: The basename of the L{FilePath} to return.
        @type path: L{str}

        @return: The sibling path.
        @rtype: L{FilePath}
        """
        return self.parent().child(path)

    def descendant(self, segments: Sequence[OtherAnyStr]) -> AbstractFilePath[OtherAnyStr]:
        """
        Retrieve a child or child's child of this path.

        @note: for type-checking, subclasses should override this signature to
            make it clear that it returns the subclass and not
            L{AbstractFilePath}.

        @param segments: A sequence of path segments as L{str} instances.

        @return: A L{FilePath} constructed by looking up the C{segments[0]}
            child of this path, the C{segments[1]} child of that path, and so
            on.

        @since: 10.2
        """
        path: AbstractFilePath[OtherAnyStr] = self
        for name in segments:
            path = path.child(name)
        return path

    def segmentsFrom(self: _Self, ancestor: _Self) -> List[AnyStr]:
        """
        Return a list of segments between a child and its ancestor.

        For example, in the case of a path X representing /a/b/c/d and a path Y
        representing /a/b, C{Y.segmentsFrom(X)} will return C{['c',
        'd']}.

        @param ancestor: an instance of the same class as self, ostensibly an
        ancestor of self.

        @raise ValueError: If the C{ancestor} parameter is not actually an
        ancestor, i.e. a path for /x/y/z is passed as an ancestor for /a/b/c/d.

        @return: a list of strs
        """
        f = self
        p: _Self = f.parent()
        segments: List[AnyStr] = []
        while f != ancestor and p != f:
            segments[0:0] = [f.basename()]
            f = p
            p = p.parent()
        if f == ancestor and segments:
            return segments
        raise ValueError(f'{ancestor!r} not parent of {self!r}')

    def __hash__(self) -> int:
        """
        Hash the same as another L{AbstractFilePath} with the same path as mine.
        """
        return hash((self.__class__, self.path))

    def getmtime(self) -> int:
        """
        Deprecated.  Use getModificationTime instead.
        """
        return int(self.getModificationTime())

    def getatime(self) -> int:
        """
        Deprecated.  Use getAccessTime instead.
        """
        return int(self.getAccessTime())

    def getctime(self) -> int:
        """
        Deprecated.  Use getStatusChangeTime instead.
        """
        return int(self.getStatusChangeTime())