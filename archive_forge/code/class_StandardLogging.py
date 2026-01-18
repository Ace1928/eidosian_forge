import logging as std_logging
import os
import fixtures
class StandardLogging(fixtures.Fixture):
    """Setup Logging redirection for tests.

    There are a number of things we want to handle with logging in tests:

    * Redirect the logging to somewhere that we can test or dump it later.

    * Ensure that as many DEBUG messages as possible are actually
       executed, to ensure they are actually syntactically valid (they
       often have not been).

    * Ensure that we create useful output for tests that doesn't
      overwhelm the testing system (which means we can't capture the
      100 MB of debug logging on every run).

    To do this we create a logger fixture at the root level, which
    defaults to INFO and create a NullLogger at DEBUG which lets
    us execute log messages at DEBUG but not keep the output.

    To support local debugging OS_DEBUG=True can be set in the
    environment, which will print out the full debug logging.

    There are also a set of overrides for particularly verbose
    modules to be even less than INFO.
    """

    def setUp(self):
        super().setUp()
        root = std_logging.getLogger()
        root.setLevel(std_logging.DEBUG)
        if os.environ.get('OS_DEBUG') in ('True', 'true', '1', 'yes'):
            level = std_logging.DEBUG
        else:
            level = std_logging.INFO
        fs = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
        self.logger = self.useFixture(fixtures.FakeLogger(format=fs, level=None))
        root.handlers[0].setLevel(level)
        if level > std_logging.DEBUG:
            handler = NullHandler()
            self.useFixture(fixtures.LogHandler(handler, nuke_handlers=False))
            handler.setLevel(std_logging.DEBUG)
            std_logging.getLogger('migrate.versioning.api').setLevel(std_logging.WARNING)
            std_logging.getLogger('alembic').setLevel(std_logging.WARNING)
            std_logging.getLogger('oslo_db.sqlalchemy').setLevel(std_logging.WARNING)

        def fake_logging_setup(*args):
            pass
        self.useFixture(fixtures.MonkeyPatch('oslo_log.log.setup', fake_logging_setup))

    def delete_stored_logs(self):
        self.logger._output.truncate(0)