import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def _consume_body_attrs(self, attrs):
    if 'provision_state' in attrs and attrs['provision_state'] is None:
        attrs['provision_state'] = 'available'
    return super(Node, self)._consume_body_attrs(attrs)