import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
class LogRateLimitTestCase(test_base.BaseTestCase):

    def tearDown(self):
        super(LogRateLimitTestCase, self).tearDown()
        rate_limit.uninstall_filter()

    def install_filter(self, *args):
        rate_limit.install_filter(*args)
        logger = logging.getLogger()

        def restore_handlers(logger, handlers):
            for handler in handlers:
                logger.addHandler(handler)
        self.addCleanup(restore_handlers, logger, list(logger.handlers))
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        return (logger, stream)

    @mock.patch('oslo_log.rate_limit.monotonic_clock')
    def test_rate_limit(self, mock_clock):
        mock_clock.return_value = 1
        logger, stream = self.install_filter(2, 1)
        logger.error('message 1')
        logger.error('message 2')
        logger.error('message 3')
        self.assertEqual(stream.getvalue(), 'message 1\nmessage 2\nLogging rate limit: drop after 2 records/1 sec\n')
        stream.seek(0)
        stream.truncate()
        mock_clock.return_value = 2
        logger.error('message 4')
        logger.error('message 5')
        logger.error('message 6')
        self.assertEqual(stream.getvalue(), 'message 4\nmessage 5\nLogging rate limit: drop after 2 records/1 sec\n')

    @mock.patch('oslo_log.rate_limit.monotonic_clock')
    def test_rate_limit_except_level(self, mock_clock):
        mock_clock.return_value = 1
        logger, stream = self.install_filter(1, 1, 'CRITICAL')
        logger.error('error 1')
        logger.error('error 2')
        logger.critical('critical 3')
        logger.critical('critical 4')
        self.assertEqual(stream.getvalue(), 'error 1\nLogging rate limit: drop after 1 records/1 sec\ncritical 3\ncritical 4\n')

    def test_install_twice(self):
        rate_limit.install_filter(100, 1)
        self.assertRaises(RuntimeError, rate_limit.install_filter, 100, 1)

    @mock.patch('oslo_log.rate_limit.monotonic_clock')
    def test_uninstall(self, mock_clock):
        mock_clock.return_value = 1
        logger, stream = self.install_filter(1, 1)
        rate_limit.uninstall_filter()
        logger.error('message 1')
        logger.error('message 2')
        logger.error('message 3')
        self.assertEqual(stream.getvalue(), 'message 1\nmessage 2\nmessage 3\n')