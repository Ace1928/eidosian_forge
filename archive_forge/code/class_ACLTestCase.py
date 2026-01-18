from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
class ACLTestCase(test_client.BaseEntityResource):

    def setUp(self):
        self._setUp('acl', entity_id='d9f95d61-8863-49d3-a045-5c2cb77130b5')
        self.secret_uuid = '8a3108ec-88fc-4f5c-86eb-f37b8ae8358e'
        self.secret_ref = self.endpoint + '/v1/secrets/' + self.secret_uuid
        self.secret_acl_ref = '{0}/acl'.format(self.secret_ref)
        self.container_uuid = '83c302c7-86fe-4f07-a277-c4962f121f19'
        self.container_ref = self.endpoint + '/v1/containers/' + self.container_uuid
        self.container_acl_ref = '{0}/acl'.format(self.container_ref)
        self.manager = self.client.acls
        self.users1 = ['2d0ee7c681cc4549b6d76769c320d91f', '721e27b8505b499e8ab3b38154705b9e']
        self.users2 = ['2d0ee7c681cc4549b6d76769c320d91f']
        self.created = str(timeutils.utcnow())

    def get_acl_response_data(self, operation_type='read', users=None, project_access=False):
        if users is None:
            users = self.users1
        op_data = {'users': users}
        op_data['project-access'] = project_access
        op_data['created'] = self.created
        op_data['updated'] = str(timeutils.utcnow())
        acl_data = {operation_type: op_data}
        return acl_data