from functools import partial
from django.contrib import auth
from django.contrib.auth import load_backend
from django.contrib.auth.backends import RemoteUserBackend
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import SimpleLazyObject
class RemoteUserMiddleware(MiddlewareMixin):
    """
    Middleware for utilizing web-server-provided authentication.

    If request.user is not authenticated, then this middleware attempts to
    authenticate the username passed in the ``REMOTE_USER`` request header.
    If authentication is successful, the user is automatically logged in to
    persist the user in the session.

    The header used is configurable and defaults to ``REMOTE_USER``.  Subclass
    this class and change the ``header`` attribute if you need to use a
    different header.
    """
    header = 'REMOTE_USER'
    force_logout_if_no_header = True

    def process_request(self, request):
        if not hasattr(request, 'user'):
            raise ImproperlyConfigured("The Django remote user auth middleware requires the authentication middleware to be installed.  Edit your MIDDLEWARE setting to insert 'django.contrib.auth.middleware.AuthenticationMiddleware' before the RemoteUserMiddleware class.")
        try:
            username = request.META[self.header]
        except KeyError:
            if self.force_logout_if_no_header and request.user.is_authenticated:
                self._remove_invalid_user(request)
            return
        if request.user.is_authenticated:
            if request.user.get_username() == self.clean_username(username, request):
                return
            else:
                self._remove_invalid_user(request)
        user = auth.authenticate(request, remote_user=username)
        if user:
            request.user = user
            auth.login(request, user)

    def clean_username(self, username, request):
        """
        Allow the backend to clean the username, if the backend defines a
        clean_username method.
        """
        backend_str = request.session[auth.BACKEND_SESSION_KEY]
        backend = auth.load_backend(backend_str)
        try:
            username = backend.clean_username(username)
        except AttributeError:
            pass
        return username

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