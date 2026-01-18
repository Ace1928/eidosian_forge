from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
class CallbackPageCollector(pb.Referenceable):
    """
    I receive pages from the peer. You may instantiate a Pager with a
    remote reference to me. I will call the callback with a list of pages
    once they are all received.
    """

    def __init__(self, callback):
        self.pages = []
        self.callback = callback

    def remote_gotPage(self, page):
        self.pages.append(page)

    def remote_endedPaging(self):
        self.callback(self.pages)