import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _get_canonical_querystring(query):
    """Generates the canonical query string given a raw query string.
    Logic is based on
    https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html

    Args:
        query (str): The raw query string.

    Returns:
        str: The canonical query string.
    """
    querystring = urllib.parse.parse_qs(query)
    querystring_encoded_map = {}
    for key in querystring:
        quote_key = urllib.parse.quote(key, safe='-_.~')
        querystring_encoded_map[quote_key] = []
        for item in querystring[key]:
            querystring_encoded_map[quote_key].append(urllib.parse.quote(item, safe='-_.~'))
        querystring_encoded_map[quote_key].sort()
    sorted_keys = list(querystring_encoded_map.keys())
    sorted_keys.sort()
    querystring_encoded_pairs = []
    for key in sorted_keys:
        for item in querystring_encoded_map[key]:
            querystring_encoded_pairs.append('{}={}'.format(key, item))
    return '&'.join(querystring_encoded_pairs)