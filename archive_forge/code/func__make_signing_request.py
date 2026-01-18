import base64
import json
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
def _make_signing_request(self, message):
    """Makes a request to the API signBlob API."""
    message = _helpers.to_bytes(message)
    method = 'POST'
    url = _SIGN_BLOB_URI.format(self._service_account_email)
    headers = {'Content-Type': 'application/json'}
    body = json.dumps({'payload': base64.b64encode(message).decode('utf-8')}).encode('utf-8')
    self._credentials.before_request(self._request, method, url, headers)
    response = self._request(url=url, method=method, body=body, headers=headers)
    if response.status != http_client.OK:
        raise exceptions.TransportError('Error calling the IAM signBlob API: {}'.format(response.data))
    return json.loads(response.data.decode('utf-8'))