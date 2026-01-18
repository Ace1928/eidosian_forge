import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPrivateFromString(self, private, type, data):
    privateKey = keys.Key.fromString(private)
    self.assertFalse(privateKey.isPublic())
    self.assertEqual(privateKey.type(), type)
    for k, v in data.items():
        self.assertEqual(privateKey.data()[k], v)