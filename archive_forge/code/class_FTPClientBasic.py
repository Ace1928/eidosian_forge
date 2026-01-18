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
class FTPClientBasic(basic.LineReceiver):
    """
    Foundations of an FTP client.
    """
    debug = False
    _encoding = 'latin-1'

    def __init__(self):
        self.actionQueue = []
        self.greeting = None
        self.nextDeferred = defer.Deferred().addCallback(self._cb_greeting)
        self.nextDeferred.addErrback(self.fail)
        self.response = []
        self._failed = 0

    def fail(self, error):
        """
        Give an error to any queued deferreds.
        """
        self._fail(error)

    def _fail(self, error):
        """
        Errback all queued deferreds.
        """
        if self._failed:
            return error
        self._failed = 1
        if self.nextDeferred:
            try:
                self.nextDeferred.errback(failure.Failure(ConnectionLost('FTP connection lost', error)))
            except defer.AlreadyCalledError:
                pass
        for ftpCommand in self.actionQueue:
            ftpCommand.fail(failure.Failure(ConnectionLost('FTP connection lost', error)))
        return error

    def _cb_greeting(self, greeting):
        self.greeting = greeting

    def sendLine(self, line):
        """
        Sends a line, unless line is None.

        @param line: Line to send
        @type line: L{bytes} or L{unicode}
        """
        if line is None:
            return
        elif isinstance(line, str):
            line = line.encode(self._encoding)
        basic.LineReceiver.sendLine(self, line)

    def sendNextCommand(self):
        """
        (Private) Processes the next command in the queue.
        """
        ftpCommand = self.popCommandQueue()
        if ftpCommand is None:
            self.nextDeferred = None
            return
        if not ftpCommand.ready:
            self.actionQueue.insert(0, ftpCommand)
            reactor.callLater(1.0, self.sendNextCommand)
            self.nextDeferred = None
            return
        if ftpCommand.text == 'PORT':
            self.generatePortCommand(ftpCommand)
        if self.debug:
            log.msg('<-- %s' % ftpCommand.text)
        self.nextDeferred = ftpCommand.deferred
        self.sendLine(ftpCommand.text)

    def queueCommand(self, ftpCommand):
        """
        Add an FTPCommand object to the queue.

        If it's the only thing in the queue, and we are connected and we aren't
        waiting for a response of an earlier command, the command will be sent
        immediately.

        @param ftpCommand: an L{FTPCommand}
        """
        self.actionQueue.append(ftpCommand)
        if len(self.actionQueue) == 1 and self.transport is not None and (self.nextDeferred is None):
            self.sendNextCommand()

    def queueStringCommand(self, command, public=1):
        """
        Queues a string to be issued as an FTP command

        @param command: string of an FTP command to queue
        @param public: a flag intended for internal use by FTPClient.  Don't
            change it unless you know what you're doing.

        @return: a L{Deferred} that will be called when the response to the
            command has been received.
        """
        ftpCommand = FTPCommand(command, public)
        self.queueCommand(ftpCommand)
        return ftpCommand.deferred

    def popCommandQueue(self):
        """
        Return the front element of the command queue, or None if empty.
        """
        if self.actionQueue:
            return self.actionQueue.pop(0)
        else:
            return None

    def queueLogin(self, username, password):
        """
        Login: send the username, send the password.

        If the password is L{None}, the PASS command won't be sent.  Also, if
        the response to the USER command has a response code of 230 (User
        logged in), then PASS won't be sent either.
        """
        deferreds = []
        userDeferred = self.queueStringCommand('USER ' + username, public=0)
        deferreds.append(userDeferred)
        if password is not None:
            passwordCmd = FTPCommand('PASS ' + password, public=0)
            self.queueCommand(passwordCmd)
            deferreds.append(passwordCmd.deferred)

            def cancelPasswordIfNotNeeded(response):
                if response[0].startswith('230'):
                    self.actionQueue.remove(passwordCmd)
                return response
            userDeferred.addCallback(cancelPasswordIfNotNeeded)
        for deferred in deferreds:
            deferred.addErrback(self.fail)
            deferred.addErrback(lambda x: None)

    def lineReceived(self, line):
        """
        (Private) Parses the response messages from the FTP server.
        """
        if bytes != str:
            line = line.decode(self._encoding)
        if self.debug:
            log.msg('--> %s' % line)
        self.response.append(line)
        codeIsValid = re.match('\\d{3} ', line)
        if not codeIsValid:
            return
        code = line[0:3]
        if code[0] == '1':
            return
        if self.nextDeferred is None:
            self.fail(UnexpectedResponse(self.response))
            return
        response = self.response
        self.response = []
        if code[0] in ('2', '3'):
            self.nextDeferred.callback(response)
        elif code[0] in ('4', '5'):
            self.nextDeferred.errback(failure.Failure(CommandFailed(response)))
        else:
            log.msg(f'Server sent invalid response code {code}')
            self.nextDeferred.errback(failure.Failure(BadResponse(response)))
        self.sendNextCommand()

    def connectionLost(self, reason):
        self._fail(reason)