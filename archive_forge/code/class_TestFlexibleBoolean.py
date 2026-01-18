import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class TestFlexibleBoolean(TestField):

    def setUp(self):
        super(TestFlexibleBoolean, self).setUp()
        self.field = fields.FlexibleBooleanField()
        self.coerce_good_values = [(True, True), (False, False), ('true', True), ('false', False), ('t', True), ('f', False), ('yes', True), ('no', False), ('y', True), ('n', False), ('on', True), ('off', False), (1, True), (0, False), ('frog', False), ('', False)]
        self.coerce_bad_values = []
        self.to_primitive_values = self.coerce_good_values[0:2]
        self.from_primitive_values = self.coerce_good_values[0:2]