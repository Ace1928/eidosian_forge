import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def _openChannel(self, channel):
    """
        Open the channel with the default connection.
        """
    self.conn.openChannel(channel)
    self.transport.packets = self.transport.packets[:-1]
    self.conn.ssh_CHANNEL_OPEN_CONFIRMATION(struct.pack('>2L', channel.id, 255) + b'\x00\x02\x00\x00\x00\x00\x80\x00')