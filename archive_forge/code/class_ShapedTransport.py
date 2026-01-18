from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class ShapedTransport(ShapedConsumer):
    """
    Wraps a C{Transport} and shapes the rate at which it receives data.

    This is a L{ShapedConsumer} with a little bit of magic to provide for
    the case where the consumer it wraps is also a C{Transport} and people
    will be attempting to access attributes this does not proxy as a
    C{Consumer} (e.g. C{loseConnection}).
    """
    iAmStreaming = False

    def __getattr__(self, name):
        return getattr(self.consumer, name)