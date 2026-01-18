import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeOAuth1Client(FakeIdentityv3Client):

    def __init__(self, **kwargs):
        super(FakeOAuth1Client, self).__init__(**kwargs)
        self.access_tokens = mock.Mock()
        self.access_tokens.resource_class = fakes.FakeResource(None, {})
        self.consumers = mock.Mock()
        self.consumers.resource_class = fakes.FakeResource(None, {})
        self.request_tokens = mock.Mock()
        self.request_tokens.resource_class = fakes.FakeResource(None, {})