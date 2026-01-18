import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
class CmdLineClientTests(ForwardingMixin, TestCase):
    """
    Connection forwarding tests run against the Conch command line client.
    """
    if runtime.platformType == 'win32':
        skip = "can't run cmdline client on win32"

    def execute(self, remoteCommand, process, sshArgs='', conchArgs=None):
        """
        As for L{OpenSSHClientTestCase.execute}, except it runs the 'conch'
        command line tool, not 'ssh'.
        """
        if conchArgs is None:
            conchArgs = []
        process.deferred = defer.Deferred()
        port = self.conchServer.getHost().port
        cmd = '-p {} -l testuser --known-hosts kh_test --user-authentications publickey -a -i dsa_test -v '.format(port) + sshArgs + ' 127.0.0.1 ' + remoteCommand
        cmds = _makeArgs(conchArgs + cmd.split())
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
        reactor.spawnProcess(process, sys.executable, encodedCmds, env=encodedEnv)
        return process.deferred

    def test_runWithLogFile(self):
        """
        It can store logs to a local file.
        """

        def cb_check_log(result):
            logContent = logPath.getContent()
            self.assertIn(b'Log opened.', logContent)
        logPath = filepath.FilePath(self.mktemp())
        d = self.execute(remoteCommand='echo goodbye', process=ConchTestOpenSSHProcess(), conchArgs=['--log', '--logfile', logPath.path, '--host-key-algorithms', 'ssh-rsa'])
        d.addCallback(self.assertEqual, b'goodbye\n')
        d.addCallback(cb_check_log)
        return d

    def test_runWithNoHostAlgorithmsSpecified(self):
        """
        Do not use --host-key-algorithms flag on command line.
        """
        d = self.execute(remoteCommand='echo goodbye', process=ConchTestOpenSSHProcess())
        d.addCallback(self.assertEqual, b'goodbye\n')
        return d