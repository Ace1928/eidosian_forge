import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def _create_properties(self):
    req = unit_test_utils.get_fake_request()
    self.properties = [(NAMESPACE3, _db_property_fixture(PROPERTY1)), (NAMESPACE3, _db_property_fixture(PROPERTY2)), (NAMESPACE1, _db_property_fixture(PROPERTY1)), (NAMESPACE6, _db_property_fixture(PROPERTY4))]
    [self.db.metadef_property_create(req.context, namespace, property) for namespace, property in self.properties]