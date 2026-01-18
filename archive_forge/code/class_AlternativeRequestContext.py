import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
class AlternativeRequestContext(object):

    def __init__(self, user=None, tenant=None):
        self.user = user
        self.tenant = tenant

    def to_dict(self):
        return {'user': self.user, 'tenant': self.tenant}