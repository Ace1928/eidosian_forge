import json
from openstack.object_store.v1 import container
from openstack.tests.unit import base
def _test_create_update(self, sot, sot_call, sess_method):
    sot.read_ACL = 'some ACL'
    sot.write_ACL = 'another ACL'
    sot.is_content_type_detected = True
    headers = {'x-container-read': 'some ACL', 'x-container-write': 'another ACL', 'x-detect-content-type': 'True', 'X-Container-Meta-foo': 'bar'}
    self.register_uris([dict(method=sess_method, uri=self.container_endpoint, json=self.body, validate=dict(headers=headers))])
    sot_call(self.cloud.object_store)
    self.assert_calls()