import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class SimpleJellyTest:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def isTheSameAs(self, other):
        return self.__dict__ == other.__dict__