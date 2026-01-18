import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class ClassA(pb.Copyable, pb.RemoteCopy):

    def __init__(self):
        self.ref = ClassB(self)