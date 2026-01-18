import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
@skipIf(runtime.platform.getType() != 'win32', 'Only runs on Windows')
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class Win32CreateProcessFlagsTests(unittest.TestCase):
    """
    Check the flags passed to CreateProcess.
    """

    @defer.inlineCallbacks
    def test_flags(self):
        """
        Verify that the flags passed to win32process.CreateProcess() prevent a
        new console window from being created. Use the following script
        to test this interactively::

            # Add the following lines to a script named
            #   should_not_open_console.pyw
            from twisted.internet import reactor, utils

            def write_result(result):
            open("output.log", "w").write(repr(result))
            reactor.stop()

            PING_EXE = r"c:\\windows\\system32\\ping.exe"
            d = utils.getProcessOutput(PING_EXE, ["slashdot.org"])
            d.addCallbacks(write_result)
            reactor.run()

        To test this, run::

            pythonw.exe should_not_open_console.pyw
        """
        from twisted.internet import _dumbwin32proc
        flags = []
        realCreateProcess = _dumbwin32proc.win32process.CreateProcess

        def fakeCreateprocess(appName, commandLine, processAttributes, threadAttributes, bInheritHandles, creationFlags, newEnvironment, currentDirectory, startupinfo):
            """
            See the Windows API documentation for I{CreateProcess} for further details.

            @param appName: The name of the module to be executed
            @param commandLine: The command line to be executed.
            @param processAttributes: Pointer to SECURITY_ATTRIBUTES structure or None.
            @param threadAttributes: Pointer to SECURITY_ATTRIBUTES structure or  None
            @param bInheritHandles: boolean to determine if inheritable handles from this
                                    process are inherited in the new process
            @param creationFlags: flags that control priority flags and creation of process.
            @param newEnvironment: pointer to new environment block for new process, or None.
            @param currentDirectory: full path to current directory of new process.
            @param startupinfo: Pointer to STARTUPINFO or STARTUPINFOEX structure
            @return: True on success, False on failure
            @rtype: L{bool}
            """
            flags.append(creationFlags)
            return realCreateProcess(appName, commandLine, processAttributes, threadAttributes, bInheritHandles, creationFlags, newEnvironment, currentDirectory, startupinfo)
        self.patch(_dumbwin32proc.win32process, 'CreateProcess', fakeCreateprocess)
        exe = sys.executable
        scriptPath = FilePath(__file__).sibling('process_cmdline.py')
        d = defer.Deferred()
        processProto = TrivialProcessProtocol(d)
        comspec = str(os.environ['COMSPEC'])
        cmd = [comspec, '/c', exe, scriptPath.path]
        _dumbwin32proc.Process(reactor, processProto, None, cmd, {}, None)
        yield d
        self.assertEqual(flags, [_dumbwin32proc.win32process.CREATE_NO_WINDOW])