import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def _construct_params(params, join_ids=True):
    """Construct/format parameter dict for an Entrez request.

    :param params: User-supplied parameters.
    :type params: dict or None
    :param bool join_ids: If True and the "id" key of ``params`` is a list
        containing multiple UIDs, join them into a single comma-delimited string.
    :returns: Parameters with defaults added and keys with None values removed.
    :rtype: dict
    """
    if params is None:
        params = {}
    params.setdefault('tool', tool)
    params.setdefault('email', email)
    params.setdefault('api_key', api_key)
    for key, value in list(params.items()):
        if value is None:
            del params[key]
    if 'email' not in params:
        warnings.warn("\n            Email address is not specified.\n\n            To make use of NCBI's E-utilities, NCBI requires you to specify your\n            email address with each request.  As an example, if your email address\n            is A.N.Other@example.com, you can specify it as follows:\n               from Bio import Entrez\n               Entrez.email = 'A.N.Other@example.com'\n            In case of excessive usage of the E-utilities, NCBI will attempt to contact\n            a user at the email address provided before blocking access to the\n            E-utilities.", UserWarning)
    if join_ids and 'id' in params:
        params['id'] = _format_ids(params['id'])
    return params