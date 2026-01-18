from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def _prepare_request_body(self, patch, prepend_key):
    body = self._body.dirty
    scheduler_hints = None
    if 'OS-SCH-HNT:scheduler_hints' in body.keys():
        scheduler_hints = body.pop('OS-SCH-HNT:scheduler_hints')
    if prepend_key and self.resource_key is not None:
        body = {self.resource_key: body}
    if scheduler_hints:
        body['OS-SCH-HNT:scheduler_hints'] = scheduler_hints
    return body