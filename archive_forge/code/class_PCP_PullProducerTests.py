from twisted.protocols import pcp
from twisted.trial import unittest
class PCP_PullProducerTests(PullProducerTest, unittest.TestCase):

    class proxyClass(pcp.BasicProducerConsumerProxy):
        iAmStreaming = False