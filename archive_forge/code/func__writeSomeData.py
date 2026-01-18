from zope.interface import implementer
from twisted.internet import interfaces
def _writeSomeData(self, data):
    """Write as much of this data as possible.

        @returns: The number of bytes written.
        """
    if self.consumer is None:
        return 0
    self.consumer.write(data)
    return len(data)