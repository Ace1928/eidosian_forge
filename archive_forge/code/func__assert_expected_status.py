import atexit
import base64
import contextlib
import datetime
import functools
import hashlib
import json
import secrets
import ldap
import os
import shutil
import socket
import sys
import uuid
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography import x509
import fixtures
import flask
from flask import testing as flask_testing
import http.client
from oslo_config import fixture as config_fixture
from oslo_context import context as oslo_context
from oslo_context import fixture as oslo_ctx_fixture
from oslo_log import fixture as log_fixture
from oslo_log import log
from oslo_utils import timeutils
import testtools
from testtools import testcase
import keystone.api
from keystone.common import context
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common as ks_ldap
from keystone import notifications
from keystone.resource.backends import base as resource_base
from keystone.server.flask import application as flask_app
from keystone.server.flask import core as keystone_flask
from keystone.tests.unit import ksfixtures
def _assert_expected_status(f):
    """Add `expected_status_code` as an argument to the test_client methods.

    `expected_status_code` must be passed as a kwarg.
    """
    TEAPOT_HTTP_STATUS = 418
    _default_expected_responses = {'get': http.client.OK, 'head': http.client.OK, 'post': http.client.CREATED, 'put': http.client.NO_CONTENT, 'patch': http.client.OK, 'delete': http.client.NO_CONTENT}

    @functools.wraps(f)
    def inner(*args, **kwargs):
        expected_status_code = kwargs.pop('expected_status_code', _default_expected_responses.get(f.__name__.lower(), http.client.OK))
        response = f(*args, **kwargs)
        if response.status_code == TEAPOT_HTTP_STATUS:
            raise AssertionError('I AM A TEAPOT(418): %s' % response.data)
        if response.status_code != expected_status_code:
            raise AssertionError('Expected HTTP Status does not match observed HTTP Status: %(expected)s != %(observed)s (%(data)s)' % {'expected': expected_status_code, 'observed': response.status_code, 'data': response.data})
        return response
    return inner