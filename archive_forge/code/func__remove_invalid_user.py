from functools import partial
from django.contrib import auth
from django.contrib.auth import load_backend
from django.contrib.auth.backends import RemoteUserBackend
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import SimpleLazyObject
def _remove_invalid_user(self, request):
    """
        Remove the current authenticated user in the request which is invalid
        but only if the user is authenticated via the RemoteUserBackend.
        """
    try:
        stored_backend = load_backend(request.session.get(auth.BACKEND_SESSION_KEY, ''))
    except ImportError:
        auth.logout(request)
    else:
        if isinstance(stored_backend, RemoteUserBackend):
            auth.logout(request)