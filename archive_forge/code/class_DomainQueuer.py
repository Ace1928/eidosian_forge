import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
class DomainQueuer:
    """
    An SMTP domain which add messages to a queue intended for relaying.
    """

    def __init__(self, service, authenticated=False):
        self.service = service
        self.authed = authenticated

    def exists(self, user):
        """
        Check whether mail can be relayed to a user.

        @type user: L{User}
        @param user: A user.

        @rtype: no-argument callable which returns L{IMessage <smtp.IMessage>}
            provider
        @return: A function which takes no arguments and returns a message
            receiver for the user.

        @raise SMTPBadRcpt: When mail cannot be relayed to the user.
        """
        if self.willRelay(user.dest, user.protocol):
            orig = filter(None, str(user.orig).split('@', 1))
            dest = filter(None, str(user.dest).split('@', 1))
            if len(orig) == 2 and len(dest) == 2:
                return lambda: self.startMessage(user)
        raise smtp.SMTPBadRcpt(user)

    def willRelay(self, address, protocol):
        """
        Check whether we agree to relay.

        The default is to relay for all connections over UNIX
        sockets and all connections from localhost.
        """
        peer = protocol.transport.getPeer()
        return self.authed or isinstance(peer, UNIXAddress) or peer.host == '127.0.0.1'

    def startMessage(self, user):
        """
        Create an envelope and a message receiver for the relay queue.

        @type user: L{User}
        @param user: A user.

        @rtype: L{IMessage <smtp.IMessage>}
        @return: A message receiver.
        """
        queue = self.service.queue
        envelopeFile, smtpMessage = queue.createNewMessage()
        with envelopeFile:
            log.msg(f'Queueing mail {str(user.orig)!r} -> {str(user.dest)!r}')
            pickle.dump([str(user.orig), str(user.dest)], envelopeFile)
        return smtpMessage