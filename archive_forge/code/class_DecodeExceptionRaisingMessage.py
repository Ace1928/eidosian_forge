from tests.unit import unittest
from boto.sqs.message import MHMessage
from boto.sqs.message import RawMessage
from boto.sqs.message import Message
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSDecodeError
from nose.plugins.attrib import attr
class DecodeExceptionRaisingMessage(RawMessage):

    @attr(sqs=True)
    def decode(self, message):
        raise SQSDecodeError('Sample decode error', self)