import json
import os
import six
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
def _fetch_certs(request, certs_url):
    """Fetches certificates.

    Google-style cerificate endpoints return JSON in the format of
    ``{'key id': 'x509 certificate'}``.

    Args:
        request (google.auth.transport.Request): The object used to make
            HTTP requests.
        certs_url (str): The certificate endpoint URL.

    Returns:
        Mapping[str, str]: A mapping of public key ID to x.509 certificate
            data.
    """
    response = request(certs_url, method='GET')
    if response.status != http_client.OK:
        raise exceptions.TransportError('Could not fetch certificates at {}'.format(certs_url))
    return json.loads(response.data.decode('utf-8'))