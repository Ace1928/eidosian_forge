from twisted.protocols import pcp
from twisted.trial import unittest
class PCP_ProducerInterfaceTests(ProducerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy