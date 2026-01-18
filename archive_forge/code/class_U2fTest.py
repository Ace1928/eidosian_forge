import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f import u2f
class U2fTest(unittest.TestCase):

    def testRegisterSuccessWithTUP(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdRegister.side_effect = [errors.TUPRequiredError, 'regdata']
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        resp = u2f_api.Register('testapp', b'ABCD', [])
        self.assertEquals(mock_sk.CmdRegister.call_count, 2)
        self.assertEquals(mock_sk.CmdWink.call_count, 1)
        self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
        self.assertEquals(resp.client_data.typ, 'navigator.id.finishEnrollment')
        self.assertEquals(resp.registration_data, 'regdata')

    def testRegisterSuccessWithPreviousKeys(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = errors.InvalidKeyHandleError
        mock_sk.CmdRegister.side_effect = [errors.TUPRequiredError, 'regdata']
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        resp = u2f_api.Register('testapp', b'ABCD', [model.RegisteredKey('khA')])
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
        self.assertTrue(mock_sk.CmdAuthenticate.call_args[0][3])
        self.assertEquals(mock_sk.CmdRegister.call_count, 2)
        self.assertEquals(mock_sk.CmdWink.call_count, 1)
        self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
        self.assertEquals(resp.client_data.typ, 'navigator.id.finishEnrollment')
        self.assertEquals(resp.registration_data, 'regdata')

    def testRegisterFailAlreadyRegistered(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = errors.TUPRequiredError
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with self.assertRaises(errors.U2FError) as cm:
            u2f_api.Register('testapp', b'ABCD', [model.RegisteredKey('khA')])
        self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
        self.assertTrue(mock_sk.CmdAuthenticate.call_args[0][3])
        self.assertEquals(mock_sk.CmdRegister.call_count, 0)
        self.assertEquals(mock_sk.CmdWink.call_count, 0)

    def testRegisterTimeout(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdRegister.side_effect = errors.TUPRequiredError
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with mock.patch.object(u2f, 'time') as _:
            with self.assertRaises(errors.U2FError) as cm:
                u2f_api.Register('testapp', b'ABCD', [])
        self.assertEquals(cm.exception.code, errors.U2FError.TIMEOUT)
        self.assertEquals(mock_sk.CmdRegister.call_count, 30)
        self.assertEquals(mock_sk.CmdWink.call_count, 30)

    def testRegisterError(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdRegister.side_effect = errors.ApduError(255, 255)
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with self.assertRaises(errors.U2FError) as cm:
            u2f_api.Register('testapp', b'ABCD', [])
        self.assertEquals(cm.exception.code, errors.U2FError.BAD_REQUEST)
        self.assertEquals(cm.exception.cause.sw1, 255)
        self.assertEquals(cm.exception.cause.sw2, 255)
        self.assertEquals(mock_sk.CmdRegister.call_count, 1)
        self.assertEquals(mock_sk.CmdWink.call_count, 0)

    def testAuthenticateSuccessWithTUP(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = [errors.TUPRequiredError, 'signature']
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        resp = u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA')])
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 2)
        self.assertEquals(mock_sk.CmdWink.call_count, 1)
        self.assertEquals(resp.key_handle, 'khA')
        self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
        self.assertEquals(resp.client_data.typ, 'navigator.id.getAssertion')
        self.assertEquals(resp.signature_data, 'signature')

    def testAuthenticateSuccessSkipInvalidKey(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = [errors.InvalidKeyHandleError, 'signature']
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        resp = u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA'), model.RegisteredKey('khB')])
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 2)
        self.assertEquals(mock_sk.CmdWink.call_count, 0)
        self.assertEquals(resp.key_handle, 'khB')
        self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
        self.assertEquals(resp.client_data.typ, 'navigator.id.getAssertion')
        self.assertEquals(resp.signature_data, 'signature')

    def testAuthenticateSuccessSkipInvalidVersion(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.return_value = 'signature'
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        resp = u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA', version='U2F_V3'), model.RegisteredKey('khB')])
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
        self.assertEquals(mock_sk.CmdWink.call_count, 0)
        self.assertEquals(resp.key_handle, 'khB')
        self.assertEquals(resp.client_data.raw_server_challenge, b'ABCD')
        self.assertEquals(resp.client_data.typ, 'navigator.id.getAssertion')
        self.assertEquals(resp.signature_data, 'signature')

    def testAuthenticateTimeout(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = errors.TUPRequiredError
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with mock.patch.object(u2f, 'time') as _:
            with self.assertRaises(errors.U2FError) as cm:
                u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA')])
        self.assertEquals(cm.exception.code, errors.U2FError.TIMEOUT)
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 30)
        self.assertEquals(mock_sk.CmdWink.call_count, 30)

    def testAuthenticateAllKeysInvalid(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = errors.InvalidKeyHandleError
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with self.assertRaises(errors.U2FError) as cm:
            u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA'), model.RegisteredKey('khB')])
        self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)
        u2f_api = u2f.U2FInterface(mock_sk)

    def testAuthenticateError(self):
        mock_sk = mock.MagicMock()
        mock_sk.CmdAuthenticate.side_effect = errors.ApduError(255, 255)
        mock_sk.CmdVersion.return_value = b'U2F_V2'
        u2f_api = u2f.U2FInterface(mock_sk)
        with self.assertRaises(errors.U2FError) as cm:
            u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA')])
        self.assertEquals(cm.exception.code, errors.U2FError.BAD_REQUEST)
        self.assertEquals(cm.exception.cause.sw1, 255)
        self.assertEquals(cm.exception.cause.sw2, 255)
        self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
        self.assertEquals(mock_sk.CmdWink.call_count, 0)