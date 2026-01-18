from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class ShapedConsumer(pcp.ProducerConsumerProxy):
    """
    Wraps a C{Consumer} and shapes the rate at which it receives data.
    """
    iAmStreaming = False

    def __init__(self, consumer, bucket):
        pcp.ProducerConsumerProxy.__init__(self, consumer)
        self.bucket = bucket
        self.bucket._refcount += 1

    def _writeSomeData(self, data):
        amount = self.bucket.add(len(data))
        return pcp.ProducerConsumerProxy._writeSomeData(self, data[:amount])

    def stopProducing(self):
        pcp.ProducerConsumerProxy.stopProducing(self)
        self.bucket._refcount -= 1