import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
class UserOAuth2(object):
    """Class to create oauth2 objects on Django request objects containing
    credentials and helper methods.
    """

    def __init__(self, request, scopes=None, return_url=None):
        """Initialize the Oauth2 Object.

        Args:
            request: Django request object.
            scopes: Scopes desired for this OAuth2 flow.
            return_url: The url to return to after the OAuth flow is complete,
                 defaults to the request's current URL path.
        """
        self.request = request
        self.return_url = return_url or request.get_full_path()
        if scopes:
            self._scopes = set(oauth2_settings.scopes) | set(scopes)
        else:
            self._scopes = set(oauth2_settings.scopes)

    def get_authorize_redirect(self):
        """Creates a URl to start the OAuth2 authorization flow."""
        get_params = {'return_url': self.return_url, 'scopes': self._get_scopes()}
        return _redirect_with_params('google_oauth:authorize', **get_params)

    def has_credentials(self):
        """Returns True if there are valid credentials for the current user
        and required scopes."""
        credentials = _credentials_from_request(self.request)
        return credentials and (not credentials.invalid) and credentials.has_scopes(self._get_scopes())

    def _get_scopes(self):
        """Returns the scopes associated with this object, kept up to
         date for incremental auth."""
        if _credentials_from_request(self.request):
            return self._scopes | _credentials_from_request(self.request).scopes
        else:
            return self._scopes

    @property
    def scopes(self):
        """Returns the scopes associated with this OAuth2 object."""
        return self._get_scopes()

    @property
    def credentials(self):
        """Gets the authorized credentials for this flow, if they exist."""
        return _credentials_from_request(self.request)

    @property
    def http(self):
        """Helper: create HTTP client authorized with OAuth2 credentials."""
        if self.has_credentials():
            return self.credentials.authorize(transport.get_http_object())
        return None