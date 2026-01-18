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
class _BaseTestCase(test.TestCase):

    def setUp(self):
        super(_BaseTestCase, self).setUp()
        self.user_id = 'fake-user'
        self.project_id = 'fake-project'
        self.context = context.RequestContext(self.user_id, self.project_id)

    def json_comparator(self, expected, obj_val):
        self.assertEqual(expected, jsonutils.dumps(obj_val))

    def str_comparator(self, expected, obj_val):
        """Compare a field to a string value

        Compare an object field to a string in the db by performing
        a simple coercion on the object field value.
        """
        self.assertEqual(expected, str(obj_val))

    def assertNotIsInstance(self, obj, cls, msg=None):
        """Python < v2.7 compatibility.  Assert 'not isinstance(obj, cls)."""
        try:
            f = super(_BaseTestCase, self).assertNotIsInstance
        except AttributeError:
            self.assertThat(obj, matchers.Not(matchers.IsInstance(cls)), message=msg or '')
        else:
            f(obj, cls, msg=msg)