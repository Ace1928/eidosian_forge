from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class HashlessFilePasswordDBMixin:
    credClass = credentials.UsernamePassword
    diskHash = None

    @staticmethod
    def networkHash(x: bytes) -> bytes:
        return x
    _validCredentials = [(b'user1', b'password1'), (b'user2', b'password2'), (b'user3', b'password3')]

    def getGoodCredentials(self):
        for u, p in self._validCredentials:
            yield (self.credClass(u, self.networkHash(p)), u)

    def getBadCredentials(self):
        for u, p in [(b'user1', b'password3'), (b'user2', b'password1'), (b'bloof', b'blarf')]:
            yield self.credClass(u, self.networkHash(p))

    def getCheckers(self):
        diskHash = self.diskHash or (lambda x: x)
        hashCheck = self.diskHash and (lambda username, password, stored: self.diskHash(password))
        for cache in (True, False):
            fn = self.mktemp()
            with open(fn, 'wb') as fObj:
                for u, p in self._validCredentials:
                    fObj.write(u + b':' + diskHash(p) + b'\n')
            yield checkers.FilePasswordDB(fn, cache=cache, hash=hashCheck)
            fn = self.mktemp()
            with open(fn, 'wb') as fObj:
                for u, p in self._validCredentials:
                    fObj.write(diskHash(p) + b' dingle dongle ' + u + b'\n')
            yield checkers.FilePasswordDB(fn, b' ', 3, 0, cache=cache, hash=hashCheck)
            fn = self.mktemp()
            with open(fn, 'wb') as fObj:
                for u, p in self._validCredentials:
                    fObj.write(b'zip,zap,' + u.title() + b',zup,' + diskHash(p) + b'\n')
            yield checkers.FilePasswordDB(fn, b',', 2, 4, False, cache=cache, hash=hashCheck)