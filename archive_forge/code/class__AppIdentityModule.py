import json
import os
import mock
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _default_async as _default
from google.auth import app_engine
from google.auth import compute_engine
from google.auth import environment_vars
from google.auth import exceptions
from google.oauth2 import _service_account_async as service_account
import google.oauth2.credentials
from tests import test__default as test_default
class _AppIdentityModule(object):
    """The interface of the App Idenity app engine module.
    See https://cloud.google.com/appengine/docs/standard/python/refdocs    /google.appengine.api.app_identity.app_identity
    """

    def get_application_id(self):
        raise NotImplementedError()