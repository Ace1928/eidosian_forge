from openstack import exceptions
from openstack import resource
def _consume_attrs(self, mapping, attrs):
    if isinstance(self, AcceleratorRequest):
        if self.resources_key in attrs:
            attrs = attrs[self.resources_key][0]
    return super(AcceleratorRequest, self)._consume_attrs(mapping, attrs)