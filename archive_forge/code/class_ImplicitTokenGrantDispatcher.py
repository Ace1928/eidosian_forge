import logging
class ImplicitTokenGrantDispatcher(Dispatcher):
    """
    This is an adapter class that will route simple Authorization Code requests,
    those that have response_type=code and a scope
    including 'openid' to either the default_grant or the oidc_grant based on
    the scopes requested.
    """

    def __init__(self, default_grant=None, oidc_grant=None):
        self.default_grant = default_grant
        self.oidc_grant = oidc_grant

    def _handler_for_request(self, request):
        handler = self.default_grant
        if request.scopes and 'openid' in request.scopes and ('id_token' in request.response_type):
            handler = self.oidc_grant
        log.debug('Selecting handler for request %r.', handler)
        return handler

    def create_authorization_response(self, request, token_handler):
        return self._handler_for_request(request).create_authorization_response(request, token_handler)

    def validate_authorization_request(self, request):
        return self._handler_for_request(request).validate_authorization_request(request)