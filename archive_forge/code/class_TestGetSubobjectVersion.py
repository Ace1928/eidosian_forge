import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class TestGetSubobjectVersion(test.TestCase):

    def setUp(self):
        super(TestGetSubobjectVersion, self).setUp()
        self.backport_mock = mock.MagicMock()
        self.rels = [('1.1', '1.0'), ('1.3', '1.1')]

    def test_get_subobject_version_not_existing(self):
        self.assertRaises(exception.TargetBeforeSubobjectExistedException, base._get_subobject_version, '1.0', self.rels, self.backport_mock)

    def test_get_subobject_version_explicit_version(self):
        base._get_subobject_version('1.3', self.rels, self.backport_mock)
        self.backport_mock.assert_called_once_with('1.1')

    def test_get_subobject_version_implicit_version(self):
        base._get_subobject_version('1.2', self.rels, self.backport_mock)
        self.backport_mock.assert_called_once_with('1.0')