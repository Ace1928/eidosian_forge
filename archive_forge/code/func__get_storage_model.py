import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def _get_storage_model():
    """This configures whether the credentials will be stored in the session
    or the Django ORM based on the settings. By default, the credentials
    will be stored in the session, unless `GOOGLE_OAUTH2_STORAGE_MODEL`
    is found in the settings. Usually, the ORM storage is used to integrate
    credentials into an existing Django user system.

    Returns:
        A tuple containing three strings, or None. If
        ``GOOGLE_OAUTH2_STORAGE_MODEL`` is configured, the tuple
        will contain the fully qualifed path of the `django.db.model`,
        the name of the ``django.contrib.auth.models.User`` field on the
        model, and the name of the
        :class:`oauth2client.contrib.django_util.models.CredentialsField`
        field on the model. If Django ORM storage is not configured,
        this function returns None.
    """
    storage_model_settings = getattr(django.conf.settings, 'GOOGLE_OAUTH2_STORAGE_MODEL', None)
    if storage_model_settings is not None:
        return (storage_model_settings['model'], storage_model_settings['user_property'], storage_model_settings['credentials_property'])
    else:
        return (None, None, None)