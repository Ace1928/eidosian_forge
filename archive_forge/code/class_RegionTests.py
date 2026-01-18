import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import regions
class RegionTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(RegionTests, self).setUp()
        self.key = 'region'
        self.collection_key = 'regions'
        self.model = regions.Region
        self.manager = self.client.regions

    def new_ref(self, **kwargs):
        kwargs = super(RegionTests, self).new_ref(**kwargs)
        kwargs.setdefault('enabled', True)
        kwargs.setdefault('id', uuid.uuid4().hex)
        return kwargs

    def test_update_enabled_defaults_to_none(self):
        super(RegionTests, self).test_update(req_ref={'description': uuid.uuid4().hex})