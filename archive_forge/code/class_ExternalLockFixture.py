import fixtures
from oslo_config import fixture as config
from oslo_concurrency import lockutils
class ExternalLockFixture(fixtures.Fixture):
    """Configure lock_path so external locks can be used in unit tests.

    Creates a temporary directory to hold file locks and sets the oslo.config
    lock_path opt to use it.  This can be used to enable external locking
    on a per-test basis, rather than globally with the OSLO_LOCK_PATH
    environment variable.

    Example::

        def test_method(self):
            self.useFixture(ExternalLockFixture())
            something_that_needs_external_locks()

    Alternatively, the useFixture call could be placed in a test class's
    setUp method to provide this functionality to all tests in the class.

    .. versionadded:: 0.3
    """

    def setUp(self):
        super(ExternalLockFixture, self).setUp()
        temp_dir = self.useFixture(fixtures.TempDir())
        conf = self.useFixture(config.Config(lockutils.CONF)).config
        conf(lock_path=temp_dir.path, group='oslo_concurrency')