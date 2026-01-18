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
def _test_test_relationships_in_order_bad(self, fake_rels):
    fake = mock.MagicMock()
    fake.VERSION = '1.5'
    fake.fields = {'foo': fields.ObjectField('bar')}
    fake.obj_relationships = fake_rels
    checker = fixture.ObjectVersionChecker()
    self.assertRaises(AssertionError, checker._test_relationships_in_order, fake)