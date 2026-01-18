from __future__ import annotations
from typing import Callable
from twisted.conch.ssh.address import SSHTransportAddress
from twisted.internet.address import IPv4Address
from twisted.internet.test.test_address import AddressTestCaseMixin
from twisted.trial import unittest
def _stringRepresentation(self, stringFunction: Callable[[object], str]) -> None:
    """
        The string representation of C{SSHTransportAddress} should be
        "SSHTransportAddress(<stringFunction on address>)".
        """
    addr = self.buildAddress()
    stringValue = stringFunction(addr)
    addressValue = stringFunction(addr.address)
    self.assertEqual(stringValue, 'SSHTransportAddress(%s)' % addressValue)