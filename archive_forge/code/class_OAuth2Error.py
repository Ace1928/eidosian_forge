from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class OAuth2Error(Exception):
    error = None
    status_code = 400
    description = ''

    def __init__(self, description=None, uri=None, state=None, status_code=None, request=None):
        """
        :param description: A human-readable ASCII [USASCII] text providing
                            additional information, used to assist the client
                            developer in understanding the error that occurred.
                            Values for the "error_description" parameter
                            MUST NOT include characters outside the set
                            x20-21 / x23-5B / x5D-7E.

        :param uri: A URI identifying a human-readable web page with information
                    about the error, used to provide the client developer with
                    additional information about the error.  Values for the
                    "error_uri" parameter MUST conform to the URI- Reference
                    syntax, and thus MUST NOT include characters outside the set
                    x21 / x23-5B / x5D-7E.

        :param state: A CSRF protection value received from the client.

        :param status_code:

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
        if description is not None:
            self.description = description
        message = '(%s) %s' % (self.error, self.description)
        if request:
            message += ' ' + repr(request)
        super(OAuth2Error, self).__init__(message)
        self.uri = uri
        self.state = state
        if status_code:
            self.status_code = status_code
        if request:
            self.redirect_uri = request.redirect_uri
            self.client_id = request.client_id
            self.scopes = request.scopes
            self.response_type = request.response_type
            self.response_mode = request.response_mode
            self.grant_type = request.grant_type
            if not state:
                self.state = request.state
        else:
            self.redirect_uri = None
            self.client_id = None
            self.scopes = None
            self.response_type = None
            self.response_mode = None
            self.grant_type = None

    def in_uri(self, uri):
        fragment = self.response_mode == 'fragment'
        return add_params_to_uri(uri, self.twotuples, fragment)

    @property
    def twotuples(self):
        error = [('error', self.error)]
        if self.description:
            error.append(('error_description', self.description))
        if self.uri:
            error.append(('error_uri', self.uri))
        if self.state:
            error.append(('state', self.state))
        return error

    @property
    def urlencoded(self):
        return urlencode(self.twotuples)

    @property
    def json(self):
        return json.dumps(dict(self.twotuples))

    @property
    def headers(self):
        if self.status_code == 401:
            '\n            https://tools.ietf.org/html/rfc6750#section-3\n\n            All challenges defined by this specification MUST use the\n            auth-scheme\n            value "Bearer".  This scheme MUST be followed by one or more\n            auth-param values.\n            '
            authvalues = ['Bearer', 'error="{}"'.format(self.error)]
            if self.description:
                authvalues.append('error_description="{}"'.format(self.description))
            if self.uri:
                authvalues.append('error_uri="{}"'.format(self.uri))
            return {'WWW-Authenticate': ', '.join(authvalues)}
        return {}