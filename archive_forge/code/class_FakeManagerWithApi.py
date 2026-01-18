import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
class FakeManagerWithApi(base.Manager):

    @api_versions.wraps('3.1')
    def return_api_version(self):
        return '3.1'

    @api_versions.wraps('3.2')
    def return_api_version(self):
        return '3.2'