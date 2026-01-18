import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def _get_scopes(self):
    """Returns the scopes associated with this object, kept up to
         date for incremental auth."""
    if _credentials_from_request(self.request):
        return self._scopes | _credentials_from_request(self.request).scopes
    else:
        return self._scopes