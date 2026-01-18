from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
class StringPager(Pager):
    """
    A simple pager that splits a string into chunks.
    """

    def __init__(self, collector, st, chunkSize=8192, callback=None, *args, **kw):
        self.string = st
        self.pointer = 0
        self.chunkSize = chunkSize
        Pager.__init__(self, collector, callback, *args, **kw)

    def nextPage(self):
        val = self.string[self.pointer:self.pointer + self.chunkSize]
        self.pointer += self.chunkSize
        if self.pointer >= len(self.string):
            self.stopPaging()
        return val