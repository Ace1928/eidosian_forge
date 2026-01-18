from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
class FakeTask(object):

    def __init__(self, task_id, type=None, status=None):
        self.task_id = task_id
        self.type = type
        self.message = None
        self.input = None
        self._status = status
        self._executor = None

    def success(self, result):
        self.result = result
        self._status = 'success'

    def fail(self, message):
        self.message = message
        self._status = 'failure'