import base64
import getpass
import json
import logging
import sys
from oauth2client.contrib import reauth_errors
from pyu2f import errors as u2ferrors
from pyu2f import model
from pyu2f.convenience import authenticator
from six.moves import urllib
class ReauthManager(object):
    """Reauth manager class that handles reauth challenges."""

    def __init__(self, http_request, access_token):
        self.http_request = http_request
        self.access_token = access_token
        self.challenges = self.InternalBuildChallenges()

    def InternalBuildChallenges(self):
        out = {}
        for c in [SecurityKeyChallenge(self.http_request, self.access_token), PasswordChallenge(self.http_request, self.access_token), SamlChallenge(self.http_request, self.access_token)]:
            if c.IsLocallyEligible():
                out[c.GetName()] = c
        return out

    def InternalStart(self, requested_scopes):
        """Does initial request to reauth API and initialize the challenges."""
        body = {'supportedChallengeTypes': list(self.challenges.keys())}
        if requested_scopes:
            body['oauthScopesForDomainPolicyLookup'] = requested_scopes
        _, content = self.http_request('{0}:start'.format(REAUTH_API), method='POST', body=json.dumps(body), headers={'Authorization': 'Bearer ' + self.access_token})
        response = json.loads(content)
        HandleErrors(response)
        return response

    def DoOneRoundOfChallenges(self, msg):
        next_msg = None
        for challenge in msg['challenges']:
            if challenge['status'] != 'READY':
                continue
            c = self.challenges[challenge['challengeType']]
            next_msg = c.Execute(challenge, msg['sessionId'])
        return next_msg

    def ObtainProofOfReauth(self, requested_scopes=None):
        """Obtain proof of reauth (rapt token)."""
        msg = None
        max_challenge_count = 5
        while max_challenge_count:
            max_challenge_count -= 1
            if not msg:
                msg = self.InternalStart(requested_scopes)
            if msg['status'] == 'AUTHENTICATED':
                return msg['encodedProofOfReauthToken']
            if not (msg['status'] == 'CHALLENGE_REQUIRED' or msg['status'] == 'CHALLENGE_PENDING'):
                raise reauth_errors.ReauthAPIError('Challenge status {0}'.format(msg['status']))
            if not InteractiveCheck():
                raise reauth_errors.ReauthUnattendedError()
            msg = self.DoOneRoundOfChallenges(msg)
        raise reauth_errors.ReauthFailError()