import logging
import shutil
import tempfile
from oslo_config import cfg
from glance_store.tests.functional import base
class TestFilesystem(base.BaseFunctionalTests):

    def __init__(self, *args, **kwargs):
        super(TestFilesystem, self).__init__('file', *args, **kwargs)

    def setUp(self):
        self.tmp_image_dir = tempfile.mkdtemp(prefix='glance_store_')
        CONF.set_override('filesystem_store_datadir', self.tmp_image_dir, group='glance_store')
        super(TestFilesystem, self).setUp()

    def tearDown(self):
        shutil.rmtree(self.tmp_image_dir)
        super(TestFilesystem, self).tearDown()