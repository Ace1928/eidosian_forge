import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
@implementer(IFTPShell)
class FTPAnonymousShell:
    """
    An anonymous implementation of IFTPShell

    @type filesystemRoot: L{twisted.python.filepath.FilePath}
    @ivar filesystemRoot: The path which is considered the root of
    this shell.
    """

    def __init__(self, filesystemRoot):
        self.filesystemRoot = filesystemRoot

    def _path(self, path):
        return self.filesystemRoot.descendant(path)

    def makeDirectory(self, path):
        return defer.fail(AnonUserDeniedError())

    def removeDirectory(self, path):
        return defer.fail(AnonUserDeniedError())

    def removeFile(self, path):
        return defer.fail(AnonUserDeniedError())

    def rename(self, fromPath, toPath):
        return defer.fail(AnonUserDeniedError())

    def receive(self, path):
        path = self._path(path)
        return defer.fail(AnonUserDeniedError())

    def openForReading(self, path):
        """
        Open C{path} for reading.

        @param path: The path, as a list of segments, to open.
        @type path: C{list} of C{unicode}
        @return: A L{Deferred} is returned that will fire with an object
            implementing L{IReadFile} if the file is successfully opened.  If
            C{path} is a directory, or if an exception is raised while trying
            to open the file, the L{Deferred} will fire with an error.
        """
        p = self._path(path)
        if p.isdir():
            return defer.fail(IsADirectoryError(path))
        try:
            f = p.open('r')
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(_FileReader(f))

    def openForWriting(self, path):
        """
        Reject write attempts by anonymous users with
        L{PermissionDeniedError}.
        """
        return defer.fail(PermissionDeniedError('STOR not allowed'))

    def access(self, path):
        p = self._path(path)
        if not p.exists():
            return defer.fail(FileNotFoundError(path))
        try:
            p.listdir()
        except OSError as e:
            return errnoToFailure(e.errno, path)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(None)

    def stat(self, path, keys=()):
        p = self._path(path)
        if p.isdir():
            try:
                statResult = self._statNode(p, keys)
            except OSError as e:
                return errnoToFailure(e.errno, path)
            except BaseException:
                return defer.fail()
            else:
                return defer.succeed(statResult)
        else:
            return self.list(path, keys).addCallback(lambda res: res[0][1])

    def list(self, path, keys=()):
        """
        Return the list of files at given C{path}, adding C{keys} stat
        informations if specified.

        @param path: the directory or file to check.
        @type path: C{str}

        @param keys: the list of desired metadata
        @type keys: C{list} of C{str}
        """
        filePath = self._path(path)
        if filePath.isdir():
            entries = filePath.listdir()
            fileEntries = [filePath.child(p) for p in entries]
        elif filePath.isfile():
            entries = [os.path.join(*filePath.segmentsFrom(self.filesystemRoot))]
            fileEntries = [filePath]
        else:
            return defer.fail(FileNotFoundError(path))
        results = []
        for fileName, filePath in zip(entries, fileEntries):
            ent = []
            results.append((fileName, ent))
            if keys:
                try:
                    ent.extend(self._statNode(filePath, keys))
                except OSError as e:
                    return errnoToFailure(e.errno, fileName)
                except BaseException:
                    return defer.fail()
        return defer.succeed(results)

    def _statNode(self, filePath, keys):
        """
        Shortcut method to get stat info on a node.

        @param filePath: the node to stat.
        @type filePath: C{filepath.FilePath}

        @param keys: the stat keys to get.
        @type keys: C{iterable}
        """
        filePath.restat()
        return [getattr(self, '_stat_' + k)(filePath) for k in keys]

    def _stat_size(self, fp):
        """
        Get the filepath's size as an int

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{int} representing the size
        """
        return fp.getsize()

    def _stat_permissions(self, fp):
        """
        Get the filepath's permissions object

        @param fp: L{twisted.python.filepath.FilePath}
        @return: L{twisted.python.filepath.Permissions} of C{fp}
        """
        return fp.getPermissions()

    def _stat_hardlinks(self, fp):
        """
        Get the number of hardlinks for the filepath - if the number of
        hardlinks is not yet implemented (say in Windows), just return 0 since
        stat-ing a file in Windows seems to return C{st_nlink=0}.

        (Reference:
        U{http://stackoverflow.com/questions/5275731/os-stat-on-windows})

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{int} representing the number of hardlinks
        """
        try:
            return fp.getNumberOfHardLinks()
        except NotImplementedError:
            return 0

    def _stat_modified(self, fp):
        """
        Get the filepath's last modified date

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{int} as seconds since the epoch
        """
        return fp.getModificationTime()

    def _stat_owner(self, fp):
        """
        Get the filepath's owner's username.  If this is not implemented
        (say in Windows) return the string "0" since stat-ing a file in
        Windows seems to return C{st_uid=0}.

        (Reference:
        U{http://stackoverflow.com/questions/5275731/os-stat-on-windows})

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{str} representing the owner's username
        """
        try:
            userID = fp.getUserID()
        except NotImplementedError:
            return '0'
        else:
            if pwd is not None:
                try:
                    return pwd.getpwuid(userID)[0]
                except KeyError:
                    pass
            return str(userID)

    def _stat_group(self, fp):
        """
        Get the filepath's owner's group.  If this is not implemented
        (say in Windows) return the string "0" since stat-ing a file in
        Windows seems to return C{st_gid=0}.

        (Reference:
        U{http://stackoverflow.com/questions/5275731/os-stat-on-windows})

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{str} representing the owner's group
        """
        try:
            groupID = fp.getGroupID()
        except NotImplementedError:
            return '0'
        else:
            if grp is not None:
                try:
                    return grp.getgrgid(groupID)[0]
                except KeyError:
                    pass
            return str(groupID)

    def _stat_directory(self, fp):
        """
        Get whether the filepath is a directory

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{bool}
        """
        return fp.isdir()