from urllib.parse import urlparse, parse_qs
from oauthlib.common import add_params_to_uri
def _non_compliant_param_name(url, headers, data):
    url_query = dict(parse_qs(urlparse(url).query))
    token = url_query.get('access_token')
    if token:
        return (url, headers, data)
    token = [('access_token', session.access_token)]
    url = add_params_to_uri(url, token)
    return (url, headers, data)