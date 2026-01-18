from __future__ import absolute_import, unicode_literals
import time
from oauthlib.common import Request, generate_token
from .. import (CONTENT_TYPE_FORM_URLENCODED, SIGNATURE_HMAC, SIGNATURE_RSA,
def _get_signature_type_and_params(self, request):
    """Extracts parameters from query, headers and body.

    Signature type
        is set to the source in which parameters were found.
        """
    header_params = signature.collect_parameters(headers=request.headers, exclude_oauth_signature=False, with_realm=True)
    body_params = signature.collect_parameters(body=request.body, exclude_oauth_signature=False)
    query_params = signature.collect_parameters(uri_query=request.uri_query, exclude_oauth_signature=False)
    params = []
    params.extend(header_params)
    params.extend(body_params)
    params.extend(query_params)
    signature_types_with_oauth_params = list(filter(lambda s: s[2], ((SIGNATURE_TYPE_AUTH_HEADER, params, utils.filter_oauth_params(header_params)), (SIGNATURE_TYPE_BODY, params, utils.filter_oauth_params(body_params)), (SIGNATURE_TYPE_QUERY, params, utils.filter_oauth_params(query_params)))))
    if len(signature_types_with_oauth_params) > 1:
        found_types = [s[0] for s in signature_types_with_oauth_params]
        raise errors.InvalidRequestError(description=('oauth_ params must come from only 1 signaturetype but were found in %s', ', '.join(found_types)))
    try:
        signature_type, params, oauth_params = signature_types_with_oauth_params[0]
    except IndexError:
        raise errors.InvalidRequestError(description='Missing mandatory OAuth parameters.')
    return (signature_type, params, oauth_params)