import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
@implementer(interfaces.IConsumer)
class _MaildirMailboxAppendMessageTask:
    """
    A task which adds a message to a maildir mailbox.

    @ivar mbox: See L{__init__}.

    @type defer: L{Deferred <defer.Deferred>} which successfully returns
        L{None}
    @ivar defer: A deferred which fires when the task has completed.

    @type opencall: L{IDelayedCall <interfaces.IDelayedCall>} provider or
        L{None}
    @ivar opencall: A scheduled call to L{prodProducer}.

    @type msg: file-like object
    @ivar msg: The message to add.

    @type tmpname: L{bytes}
    @ivar tmpname: The pathname of the temporary file holding the message while
        it is being transferred.

    @type fh: file
    @ivar fh: The new maildir file.

    @type filesender: L{FileSender <basic.FileSender>}
    @ivar filesender: A file sender which sends the message.

    @type myproducer: L{IProducer <interfaces.IProducer>}
    @ivar myproducer: The registered producer.

    @type streaming: L{bool}
    @ivar streaming: Indicates whether the registered producer provides a
        streaming interface.
    """
    osopen = staticmethod(os.open)
    oswrite = staticmethod(os.write)
    osclose = staticmethod(os.close)
    osrename = staticmethod(os.rename)

    def __init__(self, mbox, msg):
        """
        @type mbox: L{MaildirMailbox}
        @param mbox: A maildir mailbox.

        @type msg: L{bytes} or file-like object
        @param msg: The message to add.
        """
        self.mbox = mbox
        self.defer = defer.Deferred()
        self.openCall = None
        if not hasattr(msg, 'read'):
            msg = io.BytesIO(msg)
        self.msg = msg

    def startUp(self):
        """
        Start transferring the message to the mailbox.
        """
        self.createTempFile()
        if self.fh != -1:
            self.filesender = basic.FileSender()
            self.filesender.beginFileTransfer(self.msg, self)

    def registerProducer(self, producer, streaming):
        """
        Register a producer and start asking it for data if it is
        non-streaming.

        @type producer: L{IProducer <interfaces.IProducer>}
        @param producer: A producer.

        @type streaming: L{bool}
        @param streaming: A flag indicating whether the producer provides a
            streaming interface.
        """
        self.myproducer = producer
        self.streaming = streaming
        if not streaming:
            self.prodProducer()

    def prodProducer(self):
        """
        Repeatedly prod a non-streaming producer to produce data.
        """
        self.openCall = None
        if self.myproducer is not None:
            self.openCall = reactor.callLater(0, self.prodProducer)
            self.myproducer.resumeProducing()

    def unregisterProducer(self):
        """
        Finish transferring the message to the mailbox.
        """
        self.myproducer = None
        self.streaming = None
        self.osclose(self.fh)
        self.moveFileToNew()

    def write(self, data):
        """
        Write data to the maildir file.

        @type data: L{bytes}
        @param data: Data to be written to the file.
        """
        try:
            self.oswrite(self.fh, data)
        except BaseException:
            self.fail()

    def fail(self, err=None):
        """
        Fire the deferred to indicate the task completed with a failure.

        @type err: L{Failure <failure.Failure>}
        @param err: The error that occurred.
        """
        if err is None:
            err = failure.Failure()
        if self.openCall is not None:
            self.openCall.cancel()
        self.defer.errback(err)
        self.defer = None

    def moveFileToNew(self):
        """
        Place the message in the I{new/} directory, add it to the mailbox and
        fire the deferred to indicate that the task has completed
        successfully.
        """
        while True:
            newname = os.path.join(self.mbox.path, 'new', _generateMaildirName())
            try:
                self.osrename(self.tmpname, newname)
                break
            except OSError as e:
                err, estr = e.args
                import errno
                if err != errno.EEXIST:
                    self.fail()
                    newname = None
                    break
        if newname is not None:
            self.mbox.list.append(newname)
            self.defer.callback(None)
            self.defer = None

    def createTempFile(self):
        """
        Create a temporary file to hold the message as it is being transferred.
        """
        attr = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOINHERIT', 0) | getattr(os, 'O_NOFOLLOW', 0)
        tries = 0
        self.fh = -1
        while True:
            self.tmpname = os.path.join(self.mbox.path, 'tmp', _generateMaildirName())
            try:
                self.fh = self.osopen(self.tmpname, attr, 384)
                return None
            except OSError:
                tries += 1
                if tries > 500:
                    self.defer.errback(RuntimeError('Could not create tmp file for %s' % self.mbox.path))
                    self.defer = None
                    return None