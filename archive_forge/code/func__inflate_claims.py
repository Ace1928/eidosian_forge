from .exceptions import OIDCNoPrompt
import datetime
import logging
from json import loads
from oauthlib.oauth2.rfc6749.errors import ConsentRequired, InvalidRequestError, LoginRequired
def _inflate_claims(self, request):
    if request.claims and (not isinstance(request.claims, dict)):
        try:
            request.claims = loads(request.claims)
        except Exception as ex:
            raise InvalidRequestError(description='Malformed claims parameter', uri='http://openid.net/specs/openid-connect-core-1_0.html#ClaimsParameter')