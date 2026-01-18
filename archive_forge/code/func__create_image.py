import glance_store as store
import webob
import glance.api.v2.image_actions as image_actions
import glance.context
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def _create_image(self, status):
    self.images = [_db_fixture(UUID1, owner=TENANT1, checksum=CHKSUM, name='1', size=256, virtual_size=1024, visibility='public', locations=[{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}, 'status': 'active'}], disk_format='raw', container_format='bare', status=status)]
    context = self._get_fake_context()
    [self.db.image_create(context, image) for image in self.images]