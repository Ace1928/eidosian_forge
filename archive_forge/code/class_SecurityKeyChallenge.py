import abc
import base64
import sys
import pyu2f.convenience.authenticator
import pyu2f.errors
import pyu2f.model
import six
from google_reauth import _helpers, errors
class SecurityKeyChallenge(ReauthChallenge):
    """Challenge that asks for user's security key touch."""

    @property
    def name(self):
        return 'SECURITY_KEY'

    @property
    def is_locally_eligible(self):
        return True

    def obtain_challenge_input(self, metadata):
        sk = metadata['securityKey']
        challenges = sk['challenges']
        app_id = sk['applicationId']
        challenge_data = []
        for c in challenges:
            kh = c['keyHandle'].encode('ascii')
            key = pyu2f.model.RegisteredKey(bytearray(base64.urlsafe_b64decode(kh)))
            challenge = c['challenge'].encode('ascii')
            challenge = base64.urlsafe_b64decode(challenge)
            challenge_data.append({'key': key, 'challenge': challenge})
        try:
            api = pyu2f.convenience.authenticator.CreateCompositeAuthenticator(REAUTH_ORIGIN)
            response = api.Authenticate(app_id, challenge_data, print_callback=sys.stderr.write)
            return {'securityKey': response}
        except pyu2f.errors.U2FError as e:
            if e.code == pyu2f.errors.U2FError.DEVICE_INELIGIBLE:
                sys.stderr.write('Ineligible security key.\n')
            elif e.code == pyu2f.errors.U2FError.TIMEOUT:
                sys.stderr.write('Timed out while waiting for security key touch.\n')
            else:
                raise e
        except pyu2f.errors.NoDeviceFoundError:
            sys.stderr.write('No security key found.\n')
        return None