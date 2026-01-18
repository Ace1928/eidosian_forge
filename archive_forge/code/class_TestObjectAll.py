import copy
import io
from unittest import mock
from osc_lib import exceptions
from requests_mock.contrib import fixture
from openstackclient.object.v1 import object as object_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
class TestObjectAll(object_fakes.TestObjectv1):

    def setUp(self):
        super(TestObjectAll, self).setUp()
        self.requests_mock = self.useFixture(fixture.Fixture())