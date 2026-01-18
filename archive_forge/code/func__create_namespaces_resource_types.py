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
def _create_namespaces_resource_types(self):
    req = unit_test_utils.get_fake_request(roles=['admin'])
    self.ns_resource_types = [(NAMESPACE1, _db_namespace_resource_type_fixture(RESOURCE_TYPE1)), (NAMESPACE3, _db_namespace_resource_type_fixture(RESOURCE_TYPE1)), (NAMESPACE2, _db_namespace_resource_type_fixture(RESOURCE_TYPE1)), (NAMESPACE2, _db_namespace_resource_type_fixture(RESOURCE_TYPE2)), (NAMESPACE6, _db_namespace_resource_type_fixture(RESOURCE_TYPE4, prefix=PREFIX1))]
    [self.db.metadef_resource_type_association_create(req.context, namespace, ns_resource_type) for namespace, ns_resource_type in self.ns_resource_types]