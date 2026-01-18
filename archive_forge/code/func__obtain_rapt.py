from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
def _obtain_rapt(http_request, access_token, requested_scopes, rounds_num=5):
    """Given an http request method and reauth access token, get rapt token.

    Args:
        http_request: callable to run http requests. Accepts uri, method, body
            and headers. Returns a tuple: (response, content)
        access_token: reauth access token
        requested_scopes: scopes required by the client application
        rounds_num: max number of attempts to get a rapt after the next
            challenge, before failing the reauth. This defines total number of
            challenges + number of additional retries if the chalenge input
            wasn't accepted.

    Returns: rapt token.
    Raises:
        errors.ReauthError if reauth failed
    """
    msg = None
    for _ in range(0, rounds_num):
        if not msg:
            msg = _reauth_client.get_challenges(http_request, list(challenges.AVAILABLE_CHALLENGES.keys()), access_token, requested_scopes)
        if msg['status'] == _AUTHENTICATED:
            return msg['encodedProofOfReauthToken']
        if not (msg['status'] == _CHALLENGE_REQUIRED or msg['status'] == _CHALLENGE_PENDING):
            raise errors.ReauthAPIError('Challenge status {0}'.format(msg['status']))
        if not _helpers.is_interactive():
            raise errors.ReauthUnattendedError()
        msg = _run_next_challenge(msg, http_request, access_token)
    raise errors.ReauthFailError()