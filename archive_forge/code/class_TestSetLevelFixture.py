import logging
from oslo_log import fixture
from oslotest import base as test_base
class TestSetLevelFixture(test_base.BaseTestCase):

    def test_unset_before(self):
        logger = logging.getLogger('no-such-logger-unset')
        self.assertEqual(logging.NOTSET, logger.level)
        fix = fixture.SetLogLevel(['no-such-logger-unset'], logging.DEBUG)
        with fix:
            self.assertEqual(logging.DEBUG, logger.level)
        self.assertEqual(logging.NOTSET, logger.level)

    def test_set_before(self):
        logger = logging.getLogger('no-such-logger-set')
        logger.setLevel(logging.ERROR)
        self.assertEqual(logging.ERROR, logger.level)
        fix = fixture.SetLogLevel(['no-such-logger-set'], logging.DEBUG)
        with fix:
            self.assertEqual(logging.DEBUG, logger.level)
        self.assertEqual(logging.ERROR, logger.level)