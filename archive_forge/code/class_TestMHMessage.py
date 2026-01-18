from tests.unit import unittest
from boto.sqs.message import MHMessage
from boto.sqs.message import RawMessage
from boto.sqs.message import Message
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSDecodeError
from nose.plugins.attrib import attr
class TestMHMessage(unittest.TestCase):

    @attr(sqs=True)
    def test_contains(self):
        msg = MHMessage()
        msg.update({'hello': 'world'})
        self.assertTrue('hello' in msg)