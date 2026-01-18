from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def _clear_fake_version(self):
    api_versions.MAX_API_VERSION = self.orig_max
    api_versions.MIN_API_VERSION = self.orig_min