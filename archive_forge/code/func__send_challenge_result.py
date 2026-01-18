import sys
from six.moves import range
from google.auth import exceptions
from google.oauth2 import _client
from google.oauth2 import challenges
def _send_challenge_result(request, session_id, challenge_id, client_input, access_token):
    """Attempt to refresh access token by sending next challenge result.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        session_id (str): session id returned by the initial reauth call.
        challenge_id (str): challenge id returned by the initial reauth call.
        client_input: dict with a challenge-specific client input. For example:
            ``{'credential': password}`` for password challenge.
        access_token (str): Access token with reauth scopes.

    Returns:
        dict: The response from the reauth API.
    """
    body = {'sessionId': session_id, 'challengeId': challenge_id, 'action': 'RESPOND', 'proposalResponse': client_input}
    return _client._token_endpoint_request(request, _REAUTH_API + '/{}:continue'.format(session_id), body, access_token=access_token, use_json=True)