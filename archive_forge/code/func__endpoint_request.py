import json
from six.moves import urllib
from google_reauth import errors
def _endpoint_request(http_request, path, body, access_token):
    _, content = http_request(uri='{0}{1}'.format(_REAUTH_API, path), method='POST', body=json.dumps(body), headers={'Authorization': 'Bearer {0}'.format(access_token)})
    response = json.loads(content)
    _handle_errors(response)
    return response