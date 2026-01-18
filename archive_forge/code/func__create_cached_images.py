import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _create_cached_images(self):
    self.cached_images = [_db_cached_images_fixture(1, image_id=UUID1, size=256, hits=3), _db_cached_images_fixture(1, image_id=UUID3, size=1024, hits=0)]
    [self.db.insert_cache_details(None, 'node_url_1', cached_image['image_id'], cached_image['size'], hits=cached_image['hits']) for cached_image in self.cached_images]