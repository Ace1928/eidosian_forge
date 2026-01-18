import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPublicFromString(self, public, type, data):
    publicKey = keys.Key.fromString(public)
    self.assertTrue(publicKey.isPublic())
    self.assertEqual(publicKey.type(), type)
    for k, v in publicKey.data().items():
        self.assertEqual(data[k], v)