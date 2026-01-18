import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
class Auth(object):
    """Helper object that represents the auth info."""

    def __init__(self, scheme, host, project_id, public_key, secret_key=None, version=7, client=None, path='/'):
        self.scheme = scheme
        self.host = host
        self.path = path
        self.project_id = project_id
        self.public_key = public_key
        self.secret_key = secret_key
        self.version = version
        self.client = client

    @property
    def store_api_url(self):
        """Returns the API url for storing events.

        Deprecated: use get_api_url instead.
        """
        return self.get_api_url(type='store')

    def get_api_url(self, type='store'):
        """Returns the API url for storing events."""
        return '%s://%s%sapi/%s/%s/' % (self.scheme, self.host, self.path, self.project_id, type)

    def to_header(self):
        """Returns the auth header a string."""
        rv = [('sentry_key', self.public_key), ('sentry_version', self.version)]
        if self.client is not None:
            rv.append(('sentry_client', self.client))
        if self.secret_key is not None:
            rv.append(('sentry_secret', self.secret_key))
        return 'Sentry ' + ', '.join(('%s=%s' % (key, value) for key, value in rv))