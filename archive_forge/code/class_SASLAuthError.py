import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
class SASLAuthError(SASLError):
    """
    SASL Authentication failed.
    """

    def __init__(self, condition=None):
        self.condition = condition

    def __str__(self) -> str:
        return 'SASLAuthError with condition %r' % self.condition