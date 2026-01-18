from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
class FakeInherit(resource.Resource):
    a = resource.Body('a', type=Fake)