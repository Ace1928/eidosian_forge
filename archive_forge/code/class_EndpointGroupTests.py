import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import endpoint_groups
class EndpointGroupTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(EndpointGroupTests, self).setUp()
        self.key = 'endpoint_group'
        self.collection_key = 'endpoint_groups'
        self.model = endpoint_groups.EndpointGroup
        self.manager = self.client.endpoint_groups
        self.path_prefix = 'OS-EP-FILTER'

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('name', uuid.uuid4().hex)
        kwargs.setdefault('filters', '{"interface": "public"}')
        kwargs.setdefault('description', uuid.uuid4().hex)
        return kwargs