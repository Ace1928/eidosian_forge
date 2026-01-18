import base64
import copy
from datetime import datetime
import json
import six
from six.moves import http_client
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
def _make_iam_token_request(request, principal, headers, body, iam_endpoint_override=None):
    """Makes a request to the Google Cloud IAM service for an access token.
    Args:
        request (Request): The Request object to use.
        principal (str): The principal to request an access token for.
        headers (Mapping[str, str]): Map of headers to transmit.
        body (Mapping[str, str]): JSON Payload body for the iamcredentials
            API call.
        iam_endpoint_override (Optiona[str]): The full IAM endpoint override
            with the target_principal embedded. This is useful when supporting
            impersonation with regional endpoints.

    Raises:
        google.auth.exceptions.TransportError: Raised if there is an underlying
            HTTP connection error
        google.auth.exceptions.RefreshError: Raised if the impersonated
            credentials are not available.  Common reasons are
            `iamcredentials.googleapis.com` is not enabled or the
            `Service Account Token Creator` is not assigned
    """
    iam_endpoint = iam_endpoint_override or _IAM_ENDPOINT.format(principal)
    body = json.dumps(body).encode('utf-8')
    response = request(url=iam_endpoint, method='POST', headers=headers, body=body)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    if response.status != http_client.OK:
        raise exceptions.RefreshError(_REFRESH_ERROR, response_body)
    try:
        token_response = json.loads(response_body)
        token = token_response['accessToken']
        expiry = datetime.strptime(token_response['expireTime'], '%Y-%m-%dT%H:%M:%SZ')
        return (token, expiry)
    except (KeyError, ValueError) as caught_exc:
        new_exc = exceptions.RefreshError('{}: No access token or invalid expiration in response.'.format(_REFRESH_ERROR), response_body)
        six.raise_from(new_exc, caught_exc)