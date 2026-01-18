import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class TestProducer:
    """
    A dumb producer.
    """

    def __init__(self, toProduce, consumer):
        """
        @param toProduce: data to write
        @type toProduce: C{str}
        @param consumer: the consumer of data.
        @type consumer: C{IConsumer}
        """
        self.toProduce = toProduce
        self.consumer = consumer

    def start(self):
        """
        Send the data to consume.
        """
        self.consumer.write(self.toProduce)