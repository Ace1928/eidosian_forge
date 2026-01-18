from __future__ import annotations
import logging # isort:skip
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import (
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
class AuthModule(AuthProvider):
    """ An AuthProvider configured from a Python module.

    The following properties return the corresponding values from the module if
    they exist, or None otherwise:

    * ``get_login_url``,
    * ``get_user``
    * ``get_user_async``
    * ``login_url``
    * ``logout_url``

    The ``login_handler`` property will return a ``LoginHandler`` class from the
    module, or None otherwise.

    The ``logout_handler`` property will return a ``LogoutHandler`` class from
    the module, or None otherwise.

    """

    def __init__(self, module_path: PathLike) -> None:
        if not isfile(module_path):
            raise ValueError(f'no file exists at module_path: {module_path!r}')
        self._module = load_auth_module(module_path)
        super().__init__()

    @property
    def get_user(self):
        return getattr(self._module, 'get_user', None)

    @property
    def get_user_async(self):
        return getattr(self._module, 'get_user_async', None)

    @property
    def login_url(self):
        return getattr(self._module, 'login_url', None)

    @property
    def get_login_url(self):
        return getattr(self._module, 'get_login_url', None)

    @property
    def login_handler(self):
        return getattr(self._module, 'LoginHandler', None)

    @property
    def logout_url(self):
        return getattr(self._module, 'logout_url', None)

    @property
    def logout_handler(self):
        return getattr(self._module, 'LogoutHandler', None)