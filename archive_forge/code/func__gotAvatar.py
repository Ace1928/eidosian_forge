from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap
def _gotAvatar(username: bytes | tuple[()]) -> None:
    self.assertEqual(username, self.admin.username)