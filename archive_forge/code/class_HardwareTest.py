import sys
import mock
from pyu2f import errors
from pyu2f import hardware
class HardwareTest(unittest.TestCase):

    def testSimpleCommands(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        sk.CmdBlink(5)
        mock_transport.SendBlink.assert_called_once_with(5)
        sk.CmdWink()
        mock_transport.SendWink.assert_called_once_with()
        sk.CmdPing(bytearray(b'foo'))
        mock_transport.SendPing.assert_called_once_with(bytearray(b'foo'))

    def testRegisterInvalidParams(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        self.assertRaises(errors.InvalidRequestError, sk.CmdRegister, '1234', '1234')

    def testRegisterSuccess(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        mock_transport.SendMsgBytes.return_value = bytearray([1, 2, 144, 0])
        reply = sk.CmdRegister(challenge_param, app_param)
        self.assertEquals(reply, bytearray([1, 2]))
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
        self.assertEquals(sent_msg[0:4], bytearray([0, 1, 3, 0]))
        self.assertEquals(sent_msg[7:-2], bytearray(challenge_param + app_param))

    def testRegisterTUPRequired(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        mock_transport.SendMsgBytes.return_value = bytearray([105, 133])
        self.assertRaises(errors.TUPRequiredError, sk.CmdRegister, challenge_param, app_param)
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)

    def testVersion(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        mock_transport.SendMsgBytes.return_value = bytearray(b'U2F_V2\x90\x00')
        reply = sk.CmdVersion()
        self.assertEquals(reply, bytearray(b'U2F_V2'))
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
        self.assertEquals(sent_msg, bytearray([0, 3, 0, 0, 0, 0, 0]))

    def testVersionFallback(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        mock_transport.SendMsgBytes.side_effect = [bytearray([103, 0]), bytearray(b'U2F_V2\x90\x00')]
        reply = sk.CmdVersion()
        self.assertEquals(reply, bytearray(b'U2F_V2'))
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 2)
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args_list[0]
        self.assertEquals(len(sent_msg), 7)
        self.assertEquals(sent_msg[0:4], bytearray([0, 3, 0, 0]))
        self.assertEquals(sent_msg[4:7], bytearray([0, 0, 0]))
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args_list[1]
        self.assertEquals(len(sent_msg), 9)
        self.assertEquals(sent_msg[0:4], bytearray([0, 3, 0, 0]))
        self.assertEquals(sent_msg[4:7], bytearray([0, 0, 0]))
        self.assertEquals(sent_msg[7:9], bytearray([0, 0]))

    def testVersionErrors(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        mock_transport.SendMsgBytes.return_value = bytearray([250, 5])
        self.assertRaises(errors.ApduError, sk.CmdVersion)
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)

    def testAuthenticateSuccess(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        key_handle = b'\x01\x02\x03\x04'
        mock_transport.SendMsgBytes.return_value = bytearray([1, 2, 144, 0])
        reply = sk.CmdAuthenticate(challenge_param, app_param, key_handle)
        self.assertEquals(reply, bytearray([1, 2]))
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
        self.assertEquals(sent_msg[0:4], bytearray([0, 2, 3, 0]))
        self.assertEquals(sent_msg[7:-2], bytearray(challenge_param + app_param + bytearray([4, 1, 2, 3, 4])))

    def testAuthenticateCheckOnly(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        key_handle = b'\x01\x02\x03\x04'
        mock_transport.SendMsgBytes.return_value = bytearray([1, 2, 144, 0])
        reply = sk.CmdAuthenticate(challenge_param, app_param, key_handle, check_only=True)
        self.assertEquals(reply, bytearray([1, 2]))
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)
        (sent_msg,), _ = mock_transport.SendMsgBytes.call_args
        self.assertEquals(sent_msg[0:4], bytearray([0, 2, 7, 0]))
        self.assertEquals(sent_msg[7:-2], bytearray(challenge_param + app_param + bytearray([4, 1, 2, 3, 4])))

    def testAuthenticateTUPRequired(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        key_handle = b'\x01\x02\x03\x04'
        mock_transport.SendMsgBytes.return_value = bytearray([105, 133])
        self.assertRaises(errors.TUPRequiredError, sk.CmdAuthenticate, challenge_param, app_param, key_handle)
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)

    def testAuthenticateInvalidKeyHandle(self):
        mock_transport = mock.MagicMock()
        sk = hardware.SecurityKey(mock_transport)
        challenge_param = b'01234567890123456789012345678901'
        app_param = b'01234567890123456789012345678901'
        key_handle = b'\x01\x02\x03\x04'
        mock_transport.SendMsgBytes.return_value = bytearray([106, 128])
        self.assertRaises(errors.InvalidKeyHandleError, sk.CmdAuthenticate, challenge_param, app_param, key_handle)
        self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)