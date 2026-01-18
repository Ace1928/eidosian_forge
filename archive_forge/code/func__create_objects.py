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
def _create_objects(self):
    req = unit_test_utils.get_fake_request()
    self.objects = [(NAMESPACE3, _db_object_fixture(OBJECT1)), (NAMESPACE3, _db_object_fixture(OBJECT2)), (NAMESPACE1, _db_object_fixture(OBJECT1))]
    [self.db.metadef_object_create(req.context, namespace, object) for namespace, object in self.objects]