import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
class TestLegacyLogHandler(unittest.TestCase):

    def setUp(self):
        self.stream = StringIO()

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_simple_log(self):
        log = StringIO()
        with LoggingIntercept(log):
            self.handler = LogHandler(os.path.dirname(__file__), stream=self.stream, verbosity=lambda: logger.isEnabledFor(logging.DEBUG))
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)
        logger.setLevel(logging.WARNING)
        logger.info('(info)')
        self.assertEqual(self.stream.getvalue(), '')
        logger.warning('(warn)')
        ans = 'WARNING: (warn)\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.warning('(warn)')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_simple_log\n    (warn)\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_default_verbosity(self):
        log = StringIO()
        with LoggingIntercept(log):
            self.handler = LogHandler(os.path.dirname(__file__), stream=self.stream)
        self.assertIn('LogHandler class has been deprecated', log.getvalue())
        logger.addHandler(self.handler)
        logger.setLevel(logging.WARNING)
        logger.warning('(warn)')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "[base]%stest_log.py", %d, test_default_verbosity\n    (warn)\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)