from __future__ import absolute_import, unicode_literals
from oauthlib.common import extract_params, urlencode
from . import utils
def _append_params(oauth_params, params):
    """Append OAuth params to an existing set of parameters.

    Both params and oauth_params is must be lists of 2-tuples.

    Per `section 3.5.2`_ and `3.5.3`_ of the spec.

    .. _`section 3.5.2`: https://tools.ietf.org/html/rfc5849#section-3.5.2
    .. _`3.5.3`: https://tools.ietf.org/html/rfc5849#section-3.5.3

    """
    merged = list(params)
    merged.extend(oauth_params)
    merged.sort(key=lambda i: i[0].startswith('oauth_'))
    return merged