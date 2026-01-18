from __future__ import annotations
import errno
import os
import time
from typing import (
from zipfile import ZipFile
from zope.interface import implementer
from typing_extensions import Literal, Self
from twisted.python.compat import cmp, comparable
from twisted.python.filepath import (
@comparable
@implementer(IFilePath)
class ZipPath(Generic[_ZipStr, _ArchiveStr], AbstractFilePath[_ZipStr]):
    """
    I represent a file or directory contained within a zip file.
    """
    path: _ZipStr

    def __init__(self, archive: ZipArchive[_ArchiveStr], pathInArchive: _ZipStr) -> None:
        """
        Don't construct me directly.  Use C{ZipArchive.child()}.

        @param archive: a L{ZipArchive} instance.

        @param pathInArchive: a ZIP_PATH_SEP-separated string.
        """
        self.archive: ZipArchive[_ArchiveStr] = archive
        self.pathInArchive: _ZipStr = pathInArchive
        self._nativePathInArchive: _ArchiveStr = _coerceToFilesystemEncoding(archive._zipfileFilename, pathInArchive)
        sep = _coerceToFilesystemEncoding(pathInArchive, ZIP_PATH_SEP)
        archiveFilename: _ZipStr = _coerceToFilesystemEncoding(pathInArchive, archive._zipfileFilename)
        segments: List[_ZipStr] = self.pathInArchive.split(sep)
        fakePath: _ZipStr = os.path.join(archiveFilename, *segments)
        self.path: _ZipStr = fakePath

    def __cmp__(self, other: object) -> int:
        if not isinstance(other, ZipPath):
            return NotImplemented
        return cmp((self.archive, self.pathInArchive), (other.archive, other.pathInArchive))

    def __repr__(self) -> str:
        parts: List[_ZipStr]
        parts = [_coerceToFilesystemEncoding(self.sep, os.path.abspath(self.archive.path))]
        parts.extend(self.pathInArchive.split(self.sep))
        ossep = _coerceToFilesystemEncoding(self.sep, os.sep)
        return f'ZipPath({ossep.join(parts)!r})'

    @property
    def sep(self) -> _ZipStr:
        """
        Return a zip directory separator.

        @return: The zip directory separator.
        @returntype: The same type as C{self.path}.
        """
        return _coerceToFilesystemEncoding(self.path, ZIP_PATH_SEP)

    def _nativeParent(self) -> Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ArchiveStr]]:
        """
        Return parent, discarding our own encoding in favor of whatever the
        archive's is.
        """
        splitup = self.pathInArchive.split(self.sep)
        if len(splitup) == 1:
            return self.archive
        return ZipPath(self.archive, self.sep.join(splitup[:-1]))

    def parent(self) -> Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ZipStr]]:
        parent = self._nativeParent()
        if isinstance(parent, ZipArchive):
            return ZipArchive(_coerceToFilesystemEncoding(self.path, self.archive._zipfileFilename))
        return parent
    if TYPE_CHECKING:

        def parents(self) -> Iterable[Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ZipStr]]]:
            ...

    def child(self, path: OtherAnyStr) -> ZipPath[OtherAnyStr, _ArchiveStr]:
        """
        Return a new ZipPath representing a path in C{self.archive} which is
        a child of this path.

        @note: Requesting the C{".."} (or other special name) child will not
            cause L{InsecurePath} to be raised since these names do not have
            any special meaning inside a zip archive.  Be particularly
            careful with the C{path} attribute (if you absolutely must use
            it) as this means it may include special names with special
            meaning outside of the context of a zip archive.
        """
        joiner = _coerceToFilesystemEncoding(path, ZIP_PATH_SEP)
        pathInArchive = _coerceToFilesystemEncoding(path, self.pathInArchive)
        return ZipPath(self.archive, joiner.join([pathInArchive, path]))

    def sibling(self, path: OtherAnyStr) -> ZipPath[OtherAnyStr, _ArchiveStr]:
        parent: Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ZipStr]]
        rightTypedParent: Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ArchiveStr]]
        parent = self.parent()
        rightTypedParent = self.archive if isinstance(parent, ZipArchive) else parent
        child: ZipPath[OtherAnyStr, _ArchiveStr] = rightTypedParent.child(path)
        return child

    def exists(self) -> bool:
        return self.isdir() or self.isfile()

    def isdir(self) -> bool:
        return self.pathInArchive in self.archive.childmap

    def isfile(self) -> bool:
        return self.pathInArchive in self.archive.zipfile.NameToInfo

    def islink(self) -> bool:
        return False

    def listdir(self) -> List[_ZipStr]:
        if self.exists():
            if self.isdir():
                parentArchivePath: _ArchiveStr = _coerceToFilesystemEncoding(self.archive._zipfileFilename, self.pathInArchive)
                return [_coerceToFilesystemEncoding(self.path, each) for each in self.archive.childmap[parentArchivePath].keys()]
            else:
                raise UnlistableError(OSError(errno.ENOTDIR, 'Leaf zip entry listed'))
        else:
            raise UnlistableError(OSError(errno.ENOENT, 'Non-existent zip entry listed'))

    def splitext(self) -> Tuple[_ZipStr, _ZipStr]:
        """
        Return a value similar to that returned by C{os.path.splitext}.
        """
        return os.path.splitext(self.path)

    def basename(self) -> _ZipStr:
        return self.pathInArchive.split(self.sep)[-1]

    def dirname(self) -> _ZipStr:
        return self.parent().path

    def open(self, mode: Literal['r', 'w']='r') -> IO[bytes]:
        pathInArchive = _coerceToFilesystemEncoding('', self.pathInArchive)
        return self.archive.zipfile.open(pathInArchive, mode=mode)

    def changed(self) -> None:
        pass

    def getsize(self) -> int:
        """
        Retrieve this file's size.

        @return: file size, in bytes
        """
        pathInArchive = _coerceToFilesystemEncoding('', self.pathInArchive)
        return self.archive.zipfile.NameToInfo[pathInArchive].file_size

    def getAccessTime(self) -> float:
        """
        Retrieve this file's last access-time.  This is the same as the last access
        time for the archive.

        @return: a number of seconds since the epoch
        """
        return self.archive.getAccessTime()

    def getModificationTime(self) -> float:
        """
        Retrieve this file's last modification time.  This is the time of
        modification recorded in the zipfile.

        @return: a number of seconds since the epoch.
        """
        pathInArchive = _coerceToFilesystemEncoding('', self.pathInArchive)
        return time.mktime(self.archive.zipfile.NameToInfo[pathInArchive].date_time + (0, 0, 0))

    def getStatusChangeTime(self) -> float:
        """
        Retrieve this file's last modification time.  This name is provided for
        compatibility, and returns the same value as getmtime.

        @return: a number of seconds since the epoch.
        """
        return self.getModificationTime()