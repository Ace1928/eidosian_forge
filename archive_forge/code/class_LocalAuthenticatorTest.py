import base64
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import localauthenticator
@mock.patch.object(sys, 'stderr', new=mock.MagicMock())
class LocalAuthenticatorTest(unittest.TestCase):

    @mock.patch.object(localauthenticator.u2f, 'GetLocalU2FInterface')
    def testSignSuccess(self, mock_get_u2f_method):
        """Test successful signing with a valid key."""
        mock_u2f = mock.MagicMock()
        mock_get_u2f_method.return_value = mock_u2f
        mock_authenticate = mock.MagicMock()
        mock_u2f.Authenticate = mock_authenticate
        mock_authenticate.return_value = model.SignResponse(base64.urlsafe_b64decode(SIGN_SUCCESS['key_handle_encoded']), base64.urlsafe_b64decode(SIGN_SUCCESS['signature_data_encoded']), SIGN_SUCCESS['client_data'])
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = localauthenticator.LocalAuthenticator('testorigin')
        self.assertTrue(authenticator.IsAvailable())
        response = authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertTrue(mock_authenticate.called)
        authenticate_args = mock_authenticate.call_args[0]
        self.assertEqual(len(authenticate_args), 3)
        self.assertEqual(authenticate_args[0], SIGN_SUCCESS['app_id'])
        self.assertEqual(authenticate_args[1], SIGN_SUCCESS['challenge'])
        registered_keys = authenticate_args[2]
        self.assertEqual(len(registered_keys), 1)
        self.assertEqual(registered_keys[0], SIGN_SUCCESS['registered_key'])
        self.assertEquals(response.get('clientData'), SIGN_SUCCESS['client_data_encoded'])
        self.assertEquals(response.get('signatureData'), SIGN_SUCCESS['signature_data_encoded'])
        self.assertEquals(response.get('applicationId'), SIGN_SUCCESS['app_id'])
        self.assertEquals(response.get('keyHandle'), SIGN_SUCCESS['key_handle_encoded'])

    @mock.patch.object(localauthenticator.u2f, 'GetLocalU2FInterface')
    def testSignMultipleIneligible(self, mock_get_u2f_method):
        """Test signing with multiple keys registered, but none eligible."""
        mock_u2f = mock.MagicMock()
        mock_get_u2f_method.return_value = mock_u2f
        mock_authenticate = mock.MagicMock()
        mock_u2f.Authenticate = mock_authenticate
        mock_authenticate.side_effect = errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
        challenge_item = {'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}
        challenge_data = [challenge_item, challenge_item]
        authenticator = localauthenticator.LocalAuthenticator('testorigin')
        with self.assertRaises(errors.U2FError) as cm:
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)

    @mock.patch.object(localauthenticator.u2f, 'GetLocalU2FInterface')
    def testSignMultipleSuccess(self, mock_get_u2f_method):
        """Test signing with multiple keys registered and one is eligible."""
        mock_u2f = mock.MagicMock()
        mock_get_u2f_method.return_value = mock_u2f
        mock_authenticate = mock.MagicMock()
        mock_u2f.Authenticate = mock_authenticate
        return_value = model.SignResponse(base64.urlsafe_b64decode(SIGN_SUCCESS['key_handle_encoded']), base64.urlsafe_b64decode(SIGN_SUCCESS['signature_data_encoded']), SIGN_SUCCESS['client_data'])
        mock_authenticate.side_effect = [errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE), return_value]
        challenge_item = {'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}
        challenge_data = [challenge_item, challenge_item]
        authenticator = localauthenticator.LocalAuthenticator('testorigin')
        response = authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertTrue(mock_authenticate.called)
        authenticate_args = mock_authenticate.call_args[0]
        self.assertEqual(len(authenticate_args), 3)
        self.assertEqual(authenticate_args[0], SIGN_SUCCESS['app_id'])
        self.assertEqual(authenticate_args[1], SIGN_SUCCESS['challenge'])
        registered_keys = authenticate_args[2]
        self.assertEqual(len(registered_keys), 1)
        self.assertEqual(registered_keys[0], SIGN_SUCCESS['registered_key'])
        self.assertEquals(response.get('clientData'), SIGN_SUCCESS['client_data_encoded'])
        self.assertEquals(response.get('signatureData'), SIGN_SUCCESS['signature_data_encoded'])
        self.assertEquals(response.get('applicationId'), SIGN_SUCCESS['app_id'])
        self.assertEquals(response.get('keyHandle'), SIGN_SUCCESS['key_handle_encoded'])