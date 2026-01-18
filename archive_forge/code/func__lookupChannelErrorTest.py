import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def _lookupChannelErrorTest(self, code):
    """
        Deliver a request for a channel open which will result in an exception
        being raised during channel lookup.  Assert that an error response is
        delivered as a result.
        """
    self.transport.avatar._ARGS_ERROR_CODE = code
    self.conn.ssh_CHANNEL_OPEN(common.NS(b'conch-error-args') + b'\x00\x00\x00\x01' * 4)
    errors = self.flushLoggedErrors(error.ConchError)
    self.assertEqual(len(errors), 1, f'Expected one error, got: {errors!r}')
    self.assertEqual(errors[0].value.args, (123, 'error args in wrong order'))
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_OPEN_FAILURE, b'\x00\x00\x00\x01\x00\x00\x00{' + common.NS(b'error args in wrong order') + common.NS(b''))])