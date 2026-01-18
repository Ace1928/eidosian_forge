import codecs
import io
import os
import os.path
import sys
import fixtures
from oslo_config import fixture as config
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import policy
class PolicyBaseTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(PolicyBaseTestCase, self).setUp()
        self.conf = self.useFixture(config.Config()).conf
        self.config_dir = self.useFixture(fixtures.TempDir()).path
        self.conf(args=['--config-dir', self.config_dir])
        self.enforcer = policy.Enforcer(self.conf)
        self.addCleanup(self.enforcer.clear)

    def get_config_file_fullname(self, filename):
        return os.path.join(self.config_dir, filename.lstrip(os.sep))

    def create_config_file(self, filename, contents):
        """Create a configuration file under the config dir.

        Also creates any intermediate paths needed so the file can be
        in a subdirectory.

        """
        path = self.get_config_file_fullname(filename)
        pardir = os.path.dirname(path)
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        with codecs.open(path, 'w', encoding='utf-8') as f:
            f.write(contents)

    def _capture_stdout(self):
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        return sys.stdout