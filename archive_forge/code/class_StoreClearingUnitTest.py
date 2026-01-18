import os
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from glance_store._drivers import rbd as rbd_store
from glance_store._drivers import swift
from glance_store import location
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
from glance.tests import stubs
from glance.tests import utils as test_utils
class StoreClearingUnitTest(test_utils.BaseTestCase):

    def setUp(self):
        super(StoreClearingUnitTest, self).setUp()
        location.SCHEME_TO_CLS_MAP = {}
        self._create_stores()
        self.addCleanup(setattr, location, 'SCHEME_TO_CLS_MAP', dict())

    def _create_stores(self, passing_config=True):
        """Create known stores.

        :param passing_config: making store driver passes basic configurations.
        :returns: the number of how many store drivers been loaded.
        """
        store.register_opts(CONF)
        self.config(default_store='filesystem', filesystem_store_datadir=self.test_dir, group='glance_store')
        store.create_stores(CONF)