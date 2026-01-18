from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
def _scrubber_cleanup_with_store_delete_exception(self, ex):
    uri = 'file://some/path/%s' % uuid.uuid4()
    id = 'helloworldid'
    scrub = scrubber.Scrubber(glance_store)
    with patch.object(glance_store, 'delete_from_backend') as _mock_delete:
        _mock_delete.side_effect = ex
        scrub._scrub_image(id, [(id, '-', uri)])