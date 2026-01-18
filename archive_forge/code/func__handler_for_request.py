import logging
def _handler_for_request(self, request):
    handler = self.default_grant
    scopes = ()
    parameters = dict(request.decoded_body)
    client_id = parameters.get('client_id', None)
    code = parameters.get('code', None)
    redirect_uri = parameters.get('redirect_uri', None)
    if code:
        scopes = self.request_validator.get_authorization_code_scopes(client_id, code, redirect_uri, request)
    if 'openid' in scopes:
        handler = self.oidc_grant
    log.debug('Selecting handler for request %r.', handler)
    return handler