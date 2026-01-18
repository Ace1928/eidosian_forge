from __future__ import annotations
import logging # isort:skip
import importlib.util
from os.path import isfile
from types import ModuleType
from typing import (
from tornado.httputil import HTTPServerRequest
from tornado.web import RequestHandler
from ..util.serialization import make_globally_unique_id
class NullAuth(AuthProvider):
    """ A default no-auth AuthProvider.

    All of the properties of this provider return None.

    """

    @property
    def get_user(self):
        return None

    @property
    def get_user_async(self):
        return None

    @property
    def login_url(self):
        return None

    @property
    def get_login_url(self):
        return None

    @property
    def login_handler(self):
        return None

    @property
    def logout_url(self):
        return None

    @property
    def logout_handler(self):
        return None