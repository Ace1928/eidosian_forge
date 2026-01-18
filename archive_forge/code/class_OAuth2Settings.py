import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
class OAuth2Settings(object):
    """Initializes Django OAuth2 Helper Settings

    This class loads the OAuth2 Settings from the Django settings, and then
    provides those settings as attributes to the rest of the views and
    decorators in the module.

    Attributes:
      scopes: A list of OAuth2 scopes that the decorators and views will use
              as defaults.
      request_prefix: The name of the attribute that the decorators use to
                    attach the UserOAuth2 object to the Django request object.
      client_id: The OAuth2 Client ID.
      client_secret: The OAuth2 Client Secret.
    """

    def __init__(self, settings_instance):
        self.scopes = getattr(settings_instance, 'GOOGLE_OAUTH2_SCOPES', GOOGLE_OAUTH2_DEFAULT_SCOPES)
        self.request_prefix = getattr(settings_instance, 'GOOGLE_OAUTH2_REQUEST_ATTRIBUTE', GOOGLE_OAUTH2_REQUEST_ATTRIBUTE)
        info = _get_oauth2_client_id_and_secret(settings_instance)
        self.client_id, self.client_secret = info
        middleware_settings = getattr(settings_instance, 'MIDDLEWARE', None)
        if middleware_settings is None:
            middleware_settings = getattr(settings_instance, 'MIDDLEWARE_CLASSES', None)
        if middleware_settings is None:
            raise exceptions.ImproperlyConfigured('Django settings has neither MIDDLEWARE nor MIDDLEWARE_CLASSESconfigured')
        if 'django.contrib.sessions.middleware.SessionMiddleware' not in middleware_settings:
            raise exceptions.ImproperlyConfigured("The Google OAuth2 Helper requires session middleware to be installed. Edit your MIDDLEWARE_CLASSES or MIDDLEWARE setting to include 'django.contrib.sessions.middleware.SessionMiddleware'.")
        self.storage_model, self.storage_model_user_property, self.storage_model_credentials_property = _get_storage_model()