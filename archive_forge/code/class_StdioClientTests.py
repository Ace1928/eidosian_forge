import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class StdioClientTests(TestCase):
    """
    Tests for L{cftp.StdioClient}.
    """

    def setUp(self):
        """
        Create a L{cftp.StdioClient} hooked up to dummy transport and a fake
        user database.
        """
        self.fakeFilesystem = FilesystemAccessExpectations()
        sftpClient = InMemorySFTPClient(self.fakeFilesystem)
        self.client = cftp.StdioClient(sftpClient)
        self.client.currentDirectory = '/'
        self.database = self.client._pwd = UserDatabase()
        self.setKnownConsoleSize(500, 24)
        self.client.transport = self.client.client.transport

    def test_exec(self):
        """
        The I{exec} command runs its arguments locally in a child process
        using the user's shell.
        """
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', sys.executable)
        d = self.client._dispatchCommand('exec print(1 + 2)')
        d.addCallback(self.assertEqual, b'3\n')
        return d

    def test_execWithoutShell(self):
        """
        If the local user has no shell, the I{exec} command runs its arguments
        using I{/bin/sh}.
        """
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', '')
        d = self.client._dispatchCommand('exec echo hello')
        d.addCallback(self.assertEqual, b'hello\n')
        return d

    def test_bang(self):
        """
        The I{exec} command is run for lines which start with C{"!"}.
        """
        self.database.addUser(getpass.getuser(), 'secret', os.getuid(), 1234, 'foo', 'bar', '/bin/sh')
        d = self.client._dispatchCommand('!echo hello')
        d.addCallback(self.assertEqual, b'hello\n')
        return d

    def setKnownConsoleSize(self, width, height):
        """
        For the duration of this test, patch C{cftp}'s C{fcntl} module to return
        a fixed width and height.

        @param width: the width in characters
        @type width: L{int}
        @param height: the height in characters
        @type height: L{int}
        """
        import tty

        class FakeFcntl:

            def ioctl(self, fd, opt, mutate):
                if opt != tty.TIOCGWINSZ:
                    self.fail('Only window-size queries supported.')
                return struct.pack('4H', height, width, 0, 0)
        self.patch(cftp, 'fcntl', FakeFcntl())

    def test_printProgressBarReporting(self):
        """
        L{StdioClient._printProgressBar} prints a progress description,
        including percent done, amount transferred, transfer rate, and time
        remaining, all based the given start time, the given L{FileWrapper}'s
        progress information and the reactor's current time.
        """
        self.setKnownConsoleSize(10, 34)
        clock = self.client.reactor = Clock()
        wrapped = BytesIO(b'x')
        wrapped.name = b'sample'
        wrapper = cftp.FileWrapper(wrapped)
        wrapper.size = 1024 * 10
        startTime = clock.seconds()
        clock.advance(2.0)
        wrapper.total += 4096
        self.client._printProgressBar(wrapper, startTime)
        result = b"\rb'sample' 40% 4.0kB 2.0kBps 00:03 "
        self.assertEqual(self.client.transport.value(), result)

    def test_printProgressBarNoProgress(self):
        """
        L{StdioClient._printProgressBar} prints a progress description that
        indicates 0 bytes transferred if no bytes have been transferred and no
        time has passed.
        """
        self.setKnownConsoleSize(10, 34)
        clock = self.client.reactor = Clock()
        wrapped = BytesIO(b'x')
        wrapped.name = b'sample'
        wrapper = cftp.FileWrapper(wrapped)
        startTime = clock.seconds()
        self.client._printProgressBar(wrapper, startTime)
        result = b"\rb'sample'  0% 0.0B 0.0Bps 00:00 "
        self.assertEqual(self.client.transport.value(), result)

    def test_printProgressBarEmptyFile(self):
        """
        Print the progress for empty files.
        """
        self.setKnownConsoleSize(10, 34)
        wrapped = BytesIO()
        wrapped.name = b'empty-file'
        wrapper = cftp.FileWrapper(wrapped)
        self.client._printProgressBar(wrapper, 0)
        result = b"\rb'empty-file'100% 0.0B 0.0Bps 00:00 "
        self.assertEqual(result, self.client.transport.value())

    def test_getFilenameEmpty(self):
        """
        Returns empty value for both filename and remaining data.
        """
        result = self.client._getFilename('  ')
        self.assertEqual(('', ''), result)

    def test_getFilenameOnlyLocal(self):
        """
        Returns empty value for remaining data when line contains
        only a filename.
        """
        result = self.client._getFilename('only-local')
        self.assertEqual(('only-local', ''), result)

    def test_getFilenameNotQuoted(self):
        """
        Returns filename and remaining data striped of leading and trailing
        spaces.
        """
        result = self.client._getFilename(' local  remote file  ')
        self.assertEqual(('local', 'remote file'), result)

    def test_getFilenameQuoted(self):
        """
        Returns filename and remaining data not striped of leading and trailing
        spaces when quoted paths are requested.
        """
        result = self.client._getFilename(' " local file "  " remote  file " ')
        self.assertEqual((' local file ', '" remote  file "'), result)

    def makeFile(self, path=None, content=b''):
        """
        Create a local file and return its path.

        When `path` is L{None}, it will create a new temporary file.

        @param path: Optional path for the new file.
        @type path: L{str}

        @param content: Content to be written in the new file.
        @type content: L{bytes}

        @return: Path to the newly create file.
        """
        if path is None:
            path = self.mktemp()
        with open(path, 'wb') as file:
            file.write(content)
        return path

    def checkPutMessage(self, transfers, randomOrder=False):
        """
        Check output of cftp client for a put request.


        @param transfers: List with tuple of (local, remote, progress).
        @param randomOrder: When set to C{True}, it will ignore the order
            in which put reposes are received

        """
        output = self.client.transport.value()
        output = output.decode('utf-8')
        output = output.split('\n\r')
        expectedOutput = []
        actualOutput = []
        for local, remote, expected in transfers:
            expectedTransfer = []
            for line in expected:
                expectedTransfer.append(f'{local} {line}')
            expectedTransfer.append(f'Transferred {local} to {remote}')
            expectedOutput.append(expectedTransfer)
            progressParts = output.pop(0).strip('\r').split('\r')
            actual = progressParts[:-1]
            last = progressParts[-1].strip('\n').split('\n')
            actual.extend(last)
            actualTransfer = []
            for line in actual[:-1]:
                line = line.strip().rsplit(' ', 2)[0]
                line = line.strip().split(' ', 1)
                actualTransfer.append(f'{line[0]} {line[1].strip()}')
            actualTransfer.append(actual[-1])
            actualOutput.append(actualTransfer)
        if randomOrder:
            self.assertEqual(sorted(expectedOutput), sorted(actualOutput))
        else:
            self.assertEqual(expectedOutput, actualOutput)
        self.assertEqual(0, len(output), 'There are still put responses which were not checked.')

    def test_cmd_PUTSingleNoRemotePath(self):
        """
        A name based on local path is used when remote path is not
        provided.

        The progress is updated while chunks are transferred.
        """
        content = b'Test\r\nContent'
        localPath = self.makeFile(content=content)
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        remoteName = os.path.join('/', os.path.basename(localPath))
        remoteFile = InMemoryRemoteFile(remoteName)
        self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
        self.client.client.options['buffersize'] = 10
        deferred = self.client.cmd_PUT(localPath)
        self.successResultOf(deferred)
        self.assertEqual(content, remoteFile.getvalue())
        self.assertTrue(remoteFile._closed)
        self.checkPutMessage([(localPath, remoteName, ['76% 10.0B', '100% 13.0B', '100% 13.0B'])])

    def test_cmd_PUTSingleRemotePath(self):
        """
        Remote path is extracted from first filename after local file.

        Any other data in the line is ignored.
        """
        localPath = self.makeFile()
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        remoteName = '/remote-path'
        remoteFile = InMemoryRemoteFile(remoteName)
        self.fakeFilesystem.put(remoteName, flags, defer.succeed(remoteFile))
        deferred = self.client.cmd_PUT(f'{localPath} {remoteName} ignored')
        self.successResultOf(deferred)
        self.checkPutMessage([(localPath, remoteName, ['100% 0.0B'])])
        self.assertTrue(remoteFile._closed)
        self.assertEqual(b'', remoteFile.getvalue())

    def test_cmd_PUTMultipleNoRemotePath(self):
        """
        When a gobbing expression is used local files are transferred with
        remote file names based on local names.
        """
        first = self.makeFile()
        firstName = os.path.basename(first)
        secondName = 'second-name'
        parent = os.path.dirname(first)
        second = self.makeFile(path=os.path.join(parent, secondName))
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        firstRemotePath = f'/{firstName}'
        secondRemotePath = f'/{secondName}'
        firstRemoteFile = InMemoryRemoteFile(firstRemotePath)
        secondRemoteFile = InMemoryRemoteFile(secondRemotePath)
        self.fakeFilesystem.put(firstRemotePath, flags, defer.succeed(firstRemoteFile))
        self.fakeFilesystem.put(secondRemotePath, flags, defer.succeed(secondRemoteFile))
        deferred = self.client.cmd_PUT(os.path.join(parent, '*'))
        self.successResultOf(deferred)
        self.assertTrue(firstRemoteFile._closed)
        self.assertEqual(b'', firstRemoteFile.getvalue())
        self.assertTrue(secondRemoteFile._closed)
        self.assertEqual(b'', secondRemoteFile.getvalue())
        self.checkPutMessage([(first, firstRemotePath, ['100% 0.0B']), (second, secondRemotePath, ['100% 0.0B'])], randomOrder=True)

    def test_cmd_PUTMultipleWithRemotePath(self):
        """
        When a gobbing expression is used local files are transferred with
        remote file names based on local names.
        when a remote folder is requested remote paths are composed from
        remote path and local filename.
        """
        first = self.makeFile()
        firstName = os.path.basename(first)
        secondName = 'second-name'
        parent = os.path.dirname(first)
        second = self.makeFile(path=os.path.join(parent, secondName))
        flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
        firstRemoteFile = InMemoryRemoteFile(firstName)
        secondRemoteFile = InMemoryRemoteFile(secondName)
        firstRemotePath = f'/remote/{firstName}'
        secondRemotePath = f'/remote/{secondName}'
        self.fakeFilesystem.put(firstRemotePath, flags, defer.succeed(firstRemoteFile))
        self.fakeFilesystem.put(secondRemotePath, flags, defer.succeed(secondRemoteFile))
        deferred = self.client.cmd_PUT('{} remote'.format(os.path.join(parent, '*')))
        self.successResultOf(deferred)
        self.assertTrue(firstRemoteFile._closed)
        self.assertEqual(b'', firstRemoteFile.getvalue())
        self.assertTrue(secondRemoteFile._closed)
        self.assertEqual(b'', secondRemoteFile.getvalue())
        self.checkPutMessage([(first, firstName, ['100% 0.0B']), (second, secondName, ['100% 0.0B'])], randomOrder=True)