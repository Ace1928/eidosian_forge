import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
def _do_test_load_paste_app(self, expected_app_type, make_paste_file=True, paste_flavor=None, paste_config_file=None, paste_append=None):

    def _writeto(path, str):
        with open(path, 'w') as f:
            f.write(str or '')
            f.flush()

    def _appendto(orig, copy, str):
        shutil.copy(orig, copy)
        with open(copy, 'a') as f:
            f.write(str or '')
            f.flush()
    self.config(flavor=paste_flavor, config_file=paste_config_file, group='paste_deploy')
    temp_dir = self.useFixture(fixtures.TempDir()).path
    temp_file = os.path.join(temp_dir, 'testcfg.conf')
    _writeto(temp_file, '[DEFAULT]\n')
    config.parse_args(['--config-file', temp_file])
    paste_to = temp_file.replace('.conf', '-paste.ini')
    if not paste_config_file and make_paste_file:
        paste_from = os.path.join(os.getcwd(), 'etc/glance-api-paste.ini')
        _appendto(paste_from, paste_to, paste_append)
    app = config.load_paste_app('glance-api')
    self.assertIsInstance(app['/'], expected_app_type)