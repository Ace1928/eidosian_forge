from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
@implementer(IDerivedCredentials, ITestable)
class DerivedCredentials:

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def checkPassword(self, password):
        return password == self.password