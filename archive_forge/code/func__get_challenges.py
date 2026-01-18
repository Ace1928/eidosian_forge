import sys
from six.moves import range
from google.auth import exceptions
from google.oauth2 import _client
from google.oauth2 import challenges
def _get_challenges(request, supported_challenge_types, access_token, requested_scopes=None):
    """Does initial request to reauth API to get the challenges.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        supported_challenge_types (Sequence[str]): list of challenge names
            supported by the manager.
        access_token (str): Access token with reauth scopes.
        requested_scopes (Optional(Sequence[str])): Authorized scopes for the credentials.

    Returns:
        dict: The response from the reauth API.
    """
    body = {'supportedChallengeTypes': supported_challenge_types}
    if requested_scopes:
        body['oauthScopesForDomainPolicyLookup'] = requested_scopes
    return _client._token_endpoint_request(request, _REAUTH_API + ':start', body, access_token=access_token, use_json=True)