import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeFederationManager(object):

    def __init__(self, **kwargs):
        self.identity_providers = mock.Mock()
        self.identity_providers.resource_class = fakes.FakeResource(None, {})
        self.mappings = mock.Mock()
        self.mappings.resource_class = fakes.FakeResource(None, {})
        self.protocols = mock.Mock()
        self.protocols.resource_class = fakes.FakeResource(None, {})
        self.projects = mock.Mock()
        self.projects.resource_class = fakes.FakeResource(None, {})
        self.domains = mock.Mock()
        self.domains.resource_class = fakes.FakeResource(None, {})
        self.service_providers = mock.Mock()
        self.service_providers.resource_class = fakes.FakeResource(None, {})