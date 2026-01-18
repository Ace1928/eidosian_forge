import abc
import json
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
class AuthConstructor(Auth, metaclass=abc.ABCMeta):
    """Abstract base class for creating an Auth Plugin.

    The Auth Plugin created contains only one authentication method. This
    is generally the required usage.

    An AuthConstructor creates an AuthMethod based on the method's
    arguments and the auth_method_class defined by the plugin. It then
    creates the auth plugin with only that authentication method.
    """
    _auth_method_class = None

    def __init__(self, auth_url, *args, **kwargs):
        method_kwargs = self._auth_method_class._extract_kwargs(kwargs)
        method = self._auth_method_class(*args, **method_kwargs)
        super(AuthConstructor, self).__init__(auth_url, [method], **kwargs)