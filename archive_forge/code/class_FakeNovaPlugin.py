import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeNovaPlugin(object):

    @excutils.exception_filter
    def ignore_not_found(self, ex):
        if not isinstance(ex, exceptions.NotFound):
            raise ex

    def is_version_supported(self, version):
        return True

    def is_conflict(self, ex):
        return False