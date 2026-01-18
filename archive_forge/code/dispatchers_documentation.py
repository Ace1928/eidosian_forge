import logging

    This is an adapter class that will route simple Token requests, those that
    authorization_code have a scope
    including 'openid' to either the default_grant or the oidc_grant based on
    the scopes requested.
    