import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
class SMTPRelayer(RelayerMixin, smtp.SMTPClient):
    """
    A base class for SMTP relayers.
    """

    def __init__(self, messagePaths, *args, **kw):
        """
        @type messagePaths: L{list} of L{bytes}
        @param messagePaths: The base filename for each message to be relayed.

        @type args: 1-L{tuple} of (0) L{bytes} or 2-L{tuple} of
            (0) L{bytes}, (1) L{int}
        @param args: Positional arguments for L{SMTPClient.__init__}

        @type kw: L{dict}
        @param kw: Keyword arguments for L{SMTPClient.__init__}
        """
        smtp.SMTPClient.__init__(self, *args, **kw)
        self.loadMessages(messagePaths)