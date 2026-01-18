from twisted.protocols import pcp
from twisted.trial import unittest
class PCPII_ProducerInterfaceTests(ProducerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.ProducerConsumerProxy