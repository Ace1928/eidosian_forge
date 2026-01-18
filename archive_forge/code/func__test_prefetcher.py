from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
@mock.patch('glance.db.get_api')
def _test_prefetcher(self, mock_get_db):
    self.config(enabled_backends={'cheap': 'file'})
    store.register_store_opts(CONF)
    self.config(filesystem_store_datadir='/tmp', group='cheap')
    store.create_multi_stores(CONF)
    tempf = tempfile.NamedTemporaryFile()
    tempf.write(b'foo')
    db = unit_test_utils.FakeDB(initialize=False)
    mock_get_db.return_value = db
    ctx = context.RequestContext(is_admin=True, roles=['admin'])
    gateway = glance_gateway.Gateway()
    image_factory = gateway.get_image_factory(ctx)
    image_repo = gateway.get_repo(ctx)
    fetcher = prefetcher.Prefetcher()
    image = image_factory.new_image()
    image_repo.add(image)
    fetcher.cache.queue_image(image.image_id)
    self.assertFalse(fetcher.run())
    self.assertFalse(fetcher.cache.is_cached(image.image_id))
    self.assertTrue(fetcher.cache.is_queued(image.image_id))
    image.disk_format = 'raw'
    image.container_format = 'bare'
    image.status = 'active'
    loc = {'url': 'file://%s' % tempf.name, 'metadata': {'store': 'cheap'}}
    with mock.patch('glance.location._check_image_location'):
        image.locations = [loc]
    image_repo.save(image)
    self.assertTrue(fetcher.run())
    self.assertTrue(fetcher.cache.is_cached(image.image_id))
    self.assertFalse(fetcher.cache.is_queued(image.image_id))