import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
class TestLegacyPyomoFormatter(unittest.TestCase):

    def setUp(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logger.addHandler(self.handler)

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_unallowed_options(self):
        with self.assertRaisesRegex(ValueError, "'fmt' is not a valid option"):
            LegacyPyomoFormatter(fmt='%(message)')
        with self.assertRaisesRegex(ValueError, "'style' is not a valid option"):
            LegacyPyomoFormatter(style='%')

    def test_simple_log(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
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

    def test_alternate_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base='log_config'))
        logger.setLevel(logging.WARNING)
        logger.info('(info)')
        self.assertEqual(self.stream.getvalue(), '')
        logger.warning('(warn)')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_alternate_base\n    (warn)\n' % (filename, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_no_base(self):
        self.handler.setFormatter(LegacyPyomoFormatter())
        logger.setLevel(logging.WARNING)
        logger.info('(info)')
        self.assertEqual(self.stream.getvalue(), '')
        logger.warning('(warn)')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans = 'WARNING: "%s", %d, test_no_base\n    (warn)\n' % (filename, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_no_message(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        logger.setLevel(logging.WARNING)
        logger.info('')
        self.assertEqual(self.stream.getvalue(), '')
        logger.warning('')
        ans = 'WARNING:\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.warning('')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_no_message\n\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_blank_lines(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        logger.setLevel(logging.WARNING)
        logger.warning('\n\nthis is a message.\n\n\n')
        ans = 'WARNING: this is a message.\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.warning('\n\nthis is a message.\n\n\n')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_blank_lines\n    this is a message.\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_numbered_level(self):
        testname = 'test_numbered_level'
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        logger.setLevel(logging.WARNING)
        logger.log(45, '(hi)')
        ans = 'Level 45: (hi)\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.log(45, '')
        ans += 'Level 45:\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.log(45, '(hi)')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n    (hi)\n' % (os.path.sep, lineno, testname)
        self.assertEqual(self.stream.getvalue(), ans)
        logger.log(45, '')
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'Level 45: "[base]%stest_log.py", %d, %s\n\n' % (os.path.sep, lineno, testname)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_preformatted(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        msg = 'This is a long multi-line message that in normal circumstances would be line-wrapped\n        with additional information\n        that normally would be combined.'
        logger.setLevel(logging.WARNING)
        logger.info(msg)
        self.assertEqual(self.stream.getvalue(), '')
        logger.warning(Preformatted(msg))
        ans = msg + '\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.warning(msg)
        ans += 'WARNING: This is a long multi-line message that in normal circumstances would\nbe line-wrapped with additional information that normally would be combined.\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.warning(Preformatted(msg))
        ans += msg + '\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.warning(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'WARNING: "[base]%stest_log.py", %d, test_preformatted\n' % (os.path.sep, lineno)
        ans += '    This is a long multi-line message that in normal circumstances would be\n    line-wrapped with additional information that normally would be combined.\n'
        self.assertEqual(self.stream.getvalue(), ans)

    def test_long_messages(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        msg = 'This is a long message\n\nWith some kind of internal formatting\n    - including a bulleted list\n    - list 2  '
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans = 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)
        msg += '\n'
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans += 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)
        msg = '\n' + msg + '\n\n'
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans += 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
        self.assertEqual(self.stream.getvalue(), ans)
        logger.setLevel(logging.DEBUG)
        logger.info(msg)
        lineno = getframeinfo(currentframe()).lineno - 1
        ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
        self.assertEqual(self.stream.getvalue(), ans)

    def test_verbatim(self):
        self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
        msg = 'This is a long message\n\n   ```\nWith some \ninternal\nverbatim \n  - including a\n    long list\n  - and a short list \n  ```\n\nAnd some \ninternal\nnon-verbatim \n  - including a\n    long list\n  - and a short list \n\nAnd a section\n~~~~~~~~~~~~~\n\n  | and\n  | a line\n  | block\n\nAnd a\nquoted literal::\n\n>> he said\n>\n> and they replied\n\nthis is\noutside the quote\n\nindented literal::\n\n    Here is\n       an indented\n\n    literal\n    with a blank line\n\nFinally, an invalid::\n\nquote\nblock\n'
        logger.setLevel(logging.WARNING)
        logger.warning(msg)
        ans = 'WARNING: This is a long message\n\n    With some \n    internal\n    verbatim \n      - including a\n        long list\n      - and a short list \n\n    And some internal non-verbatim\n      - including a long list\n      - and a short list\n\n    And a section\n    ~~~~~~~~~~~~~\n\n      | and\n      | a line\n      | block\n\n    And a quoted literal::\n\n    >> he said\n    >\n    > and they replied\n\n    this is outside the quote\n\n    indented literal::\n\n        Here is\n           an indented\n\n        literal\n        with a blank line\n\n    Finally, an invalid::\n\n    quote block\n'
        self.maxDiff = None
        self.assertEqual(self.stream.getvalue(), ans)