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
class OurServerCmdLineClientTests(CFTPClientTestBase):
    """
    Functional tests which launch a SFTP server over TCP on localhost and check
    cftp command line interface using a spawned process.

    Due to the spawned process you can not add a debugger breakpoint for the
    client code.
    """

    def setUp(self):
        CFTPClientTestBase.setUp(self)
        self.startServer()
        cmds = '-p %i -l testuser --known-hosts kh_test --user-authentications publickey --host-key-algorithms ssh-rsa -i dsa_test -a -v 127.0.0.1'
        port = self.server.getHost().port
        cmds = test_conch._makeArgs((cmds % port).split(), mod='cftp')
        log.msg(f'running {sys.executable} {cmds}')
        d = defer.Deferred()
        self.processProtocol = SFTPTestProcess(d)
        d.addCallback(lambda _: self.processProtocol.clearBuffer())
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        encodedCmds = []
        encodedEnv = {}
        for cmd in cmds:
            if isinstance(cmd, str):
                cmd = cmd.encode('utf-8')
            encodedCmds.append(cmd)
        for var in env:
            val = env[var]
            if isinstance(var, str):
                var = var.encode('utf-8')
            if isinstance(val, str):
                val = val.encode('utf-8')
            encodedEnv[var] = val
        log.msg(encodedCmds)
        log.msg(encodedEnv)
        reactor.spawnProcess(self.processProtocol, sys.executable, encodedCmds, env=encodedEnv)
        return d

    def tearDown(self):
        d = self.stopServer()
        d.addCallback(lambda _: self.processProtocol.killProcess())
        return d

    def _killProcess(self, ignored):
        try:
            self.processProtocol.transport.signalProcess('KILL')
        except error.ProcessExitedAlready:
            pass

    def runCommand(self, command):
        """
        Run the given command with the cftp client. Return a C{Deferred} that
        fires when the command is complete. Payload is the server's output for
        that command.
        """
        return self.processProtocol.runCommand(command)

    def runScript(self, *commands):
        """
        Run the given commands with the cftp client. Returns a C{Deferred}
        that fires when the commands are all complete. The C{Deferred}'s
        payload is a list of output for each command.
        """
        return self.processProtocol.runScript(commands)

    def testCdPwd(self):
        """
        Test that 'pwd' reports the current remote directory, that 'lpwd'
        reports the current local directory, and that changing to a
        subdirectory then changing to its parent leaves you in the original
        remote directory.
        """
        homeDir = self.testDir
        d = self.runScript('pwd', 'lpwd', 'cd testDirectory', 'cd ..', 'pwd')

        def cmdOutput(output):
            """
            Callback function for handling command output.
            """
            cmds = []
            for cmd in output:
                if isinstance(cmd, bytes):
                    cmd = cmd.decode('utf-8')
                cmds.append(cmd)
            return cmds[:3] + cmds[4:]
        d.addCallback(cmdOutput)
        d.addCallback(self.assertEqual, [homeDir.path, os.getcwd(), '', homeDir.path])
        return d

    def testChAttrs(self):
        """
        Check that 'ls -l' output includes the access permissions and that
        this output changes appropriately with 'chmod'.
        """

        def _check(results):
            self.flushLoggedErrors()
            self.assertTrue(results[0].startswith(b'-rw-r--r--'))
            self.assertEqual(results[1], b'')
            self.assertTrue(results[2].startswith(b'----------'), results[2])
            self.assertEqual(results[3], b'')
        d = self.runScript('ls -l testfile1', 'chmod 0 testfile1', 'ls -l testfile1', 'chmod 644 testfile1')
        return d.addCallback(_check)

    def testList(self):
        """
        Check 'ls' works as expected. Checks for wildcards, hidden files,
        listing directories and listing empty directories.
        """

        def _check(results):
            self.assertEqual(results[0], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
            self.assertEqual(results[1], [b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])
            self.assertEqual(results[2], [b'testRemoveFile', b'testRenameFile'])
            self.assertEqual(results[3], [b'.testHiddenFile', b'testRemoveFile', b'testRenameFile'])
            self.assertEqual(results[4], [b''])
        d = self.runScript('ls', 'ls ../' + self.testDir.basename(), 'ls *File', 'ls -a *File', 'ls -l testDirectory')
        d.addCallback(lambda xs: [x.split(b'\n') for x in xs])
        return d.addCallback(_check)

    def testHelp(self):
        """
        Check that running the '?' command returns help.
        """
        d = self.runCommand('?')
        helpText = cftp.StdioClient(None).cmd_HELP('').strip()
        if isinstance(helpText, str):
            helpText = helpText.encode('utf-8')
        d.addCallback(self.assertEqual, helpText)
        return d

    def assertFilesEqual(self, name1, name2, msg=None):
        """
        Assert that the files at C{name1} and C{name2} contain exactly the
        same data.
        """
        self.assertEqual(name1.getContent(), name2.getContent(), msg)

    def testGet(self):
        """
        Test that 'get' saves the remote file to the correct local location,
        that the output of 'get' is correct and that 'rm' actually removes
        the file.
        """
        expectedOutput = 'Transferred {}/testfile1 to {}/test file2'.format(self.testDir.path, self.testDir.path)
        if isinstance(expectedOutput, str):
            expectedOutput = expectedOutput.encode('utf-8')

        def _checkGet(result):
            self.assertTrue(result.endswith(expectedOutput))
            self.assertFilesEqual(self.testDir.child('testfile1'), self.testDir.child('test file2'), 'get failed')
            return self.runCommand('rm "test file2"')
        d = self.runCommand(f'get testfile1 "{self.testDir.path}/test file2"')
        d.addCallback(_checkGet)
        d.addCallback(lambda _: self.assertFalse(self.testDir.child('test file2').exists()))
        return d

    def testWildcardGet(self):
        """
        Test that 'get' works correctly when given wildcard parameters.
        """

        def _check(ignored):
            self.assertFilesEqual(self.testDir.child('testRemoveFile'), FilePath('testRemoveFile'), 'testRemoveFile get failed')
            self.assertFilesEqual(self.testDir.child('testRenameFile'), FilePath('testRenameFile'), 'testRenameFile get failed')
        d = self.runCommand('get testR*')
        return d.addCallback(_check)

    def testPut(self):
        """
        Check that 'put' uploads files correctly and that they can be
        successfully removed. Also check the output of the put command.
        """
        expectedOutput = b'Transferred ' + self.testDir.asBytesMode().path + b'/testfile1 to ' + self.testDir.asBytesMode().path + b'/test"file2'

        def _checkPut(result):
            self.assertFilesEqual(self.testDir.child('testfile1'), self.testDir.child('test"file2'))
            self.assertTrue(result.endswith(expectedOutput))
            return self.runCommand('rm "test\\"file2"')
        d = self.runCommand(f'put {self.testDir.path}/testfile1 "test\\"file2"')
        d.addCallback(_checkPut)
        d.addCallback(lambda _: self.assertFalse(self.testDir.child('test"file2').exists()))
        return d

    def test_putOverLongerFile(self):
        """
        Check that 'put' uploads files correctly when overwriting a longer
        file.
        """
        with self.testDir.child('shorterFile').open(mode='w') as f:
            f.write(b'a')
        with self.testDir.child('longerFile').open(mode='w') as f:
            f.write(b'bb')

        def _checkPut(result):
            self.assertFilesEqual(self.testDir.child('shorterFile'), self.testDir.child('longerFile'))
        d = self.runCommand(f'put {self.testDir.path}/shorterFile longerFile')
        d.addCallback(_checkPut)
        return d

    def test_putMultipleOverLongerFile(self):
        """
        Check that 'put' uploads files correctly when overwriting a longer
        file and you use a wildcard to specify the files to upload.
        """
        someDir = self.testDir.child('dir')
        someDir.createDirectory()
        with someDir.child('file').open(mode='w') as f:
            f.write(b'a')
        with self.testDir.child('file').open(mode='w') as f:
            f.write(b'bb')

        def _checkPut(result):
            self.assertFilesEqual(someDir.child('file'), self.testDir.child('file'))
        d = self.runCommand(f'put {self.testDir.path}/dir/*')
        d.addCallback(_checkPut)
        return d

    def testWildcardPut(self):
        """
        What happens if you issue a 'put' command and include a wildcard (i.e.
        '*') in parameter? Check that all files matching the wildcard are
        uploaded to the correct directory.
        """

        def check(results):
            self.assertEqual(results[0], b'')
            self.assertEqual(results[2], b'')
            self.assertFilesEqual(self.testDir.child('testRemoveFile'), self.testDir.parent().child('testRemoveFile'), 'testRemoveFile get failed')
            self.assertFilesEqual(self.testDir.child('testRenameFile'), self.testDir.parent().child('testRenameFile'), 'testRenameFile get failed')
        d = self.runScript('cd ..', f'put {self.testDir.path}/testR*', 'cd %s' % self.testDir.basename())
        d.addCallback(check)
        return d

    def testLink(self):
        """
        Test that 'ln' creates a file which appears as a link in the output of
        'ls'. Check that removing the new file succeeds without output.
        """

        def _check(results):
            self.flushLoggedErrors()
            self.assertEqual(results[0], b'')
            self.assertTrue(results[1].startswith(b'l'), 'link failed')
            return self.runCommand('rm testLink')
        d = self.runScript('ln testLink testfile1', 'ls -l testLink')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d

    def testRemoteDirectory(self):
        """
        Test that we can create and remove directories with the cftp client.
        """

        def _check(results):
            self.assertEqual(results[0], b'')
            self.assertTrue(results[1].startswith(b'd'))
            return self.runCommand('rmdir testMakeDirectory')
        d = self.runScript('mkdir testMakeDirectory', 'ls -l testMakeDirector?')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d

    def test_existingRemoteDirectory(self):
        """
        Test that a C{mkdir} on an existing directory fails with the
        appropriate error, and doesn't log an useless error server side.
        """

        def _check(results):
            self.assertEqual(results[0], b'')
            self.assertEqual(results[1], b'remote error 11: mkdir failed')
        d = self.runScript('mkdir testMakeDirectory', 'mkdir testMakeDirectory')
        d.addCallback(_check)
        return d

    def testLocalDirectory(self):
        """
        Test that we can create a directory locally and remove it with the
        cftp client. This test works because the 'remote' server is running
        out of a local directory.
        """
        d = self.runCommand(f'lmkdir {self.testDir.path}/testLocalDirectory')
        d.addCallback(self.assertEqual, b'')
        d.addCallback(lambda _: self.runCommand('rmdir testLocalDirectory'))
        d.addCallback(self.assertEqual, b'')
        return d

    def testRename(self):
        """
        Test that we can rename a file.
        """

        def _check(results):
            self.assertEqual(results[0], b'')
            self.assertEqual(results[1], b'testfile2')
            return self.runCommand('rename testfile2 testfile1')
        d = self.runScript('rename testfile1 testfile2', 'ls testfile?')
        d.addCallback(_check)
        d.addCallback(self.assertEqual, b'')
        return d