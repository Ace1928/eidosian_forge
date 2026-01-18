import logging
import fixtures
class SetLogLevel(fixtures.Fixture):
    """Override the log level for the named loggers, restoring their
    previous value at the end of the test.

    To use::

      from oslo_log import fixture as log_fixture

      self.useFixture(log_fixture.SetLogLevel(['myapp.foo'], logging.DEBUG))

    :param logger_names: Sequence of logger names, as would be passed
                         to getLogger().
    :type logger_names: list(str)
    :param level: Logging level, usually one of logging.DEBUG,
                  logging.INFO, etc.
    :type level: int
    """

    def __init__(self, logger_names, level):
        self.logger_names = logger_names
        self.level = level

    def setUp(self):
        super(SetLogLevel, self).setUp()
        for name in self.logger_names:
            logger = logging.getLogger(name)
            self.addCleanup(logger.setLevel, logger.level)
            logger.setLevel(self.level)