import contextlib
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from tornado.escape import utf8
from tornado.log import LogFormatter, define_logging_options, enable_pretty_logging
from tornado.options import OptionParser
from tornado.util import basestring_type
class LogFormatterTest(unittest.TestCase):
    LINE_RE = re.compile(b'(?s)\x01\\[E [0-9]{6} [0-9]{2}:[0-9]{2}:[0-9]{2} log_test:[0-9]+\\]\x02 (.*)')

    def setUp(self):
        self.formatter = LogFormatter(color=False)
        self.formatter._colors = {logging.ERROR: '\x01'}
        self.formatter._normal = '\x02'
        self.logger = logging.Logger('LogFormatterTest')
        self.logger.propagate = False
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, 'log.out')
        self.handler = self.make_handler(self.filename)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.handler.close()
        os.unlink(self.filename)
        os.rmdir(self.tempdir)

    def make_handler(self, filename):
        return logging.FileHandler(filename, encoding='utf-8')

    def get_output(self):
        with open(self.filename, 'rb') as f:
            line = f.read().strip()
            m = LogFormatterTest.LINE_RE.match(line)
            if m:
                return m.group(1)
            else:
                raise Exception("output didn't match regex: %r" % line)

    def test_basic_logging(self):
        self.logger.error('foo')
        self.assertEqual(self.get_output(), b'foo')

    def test_bytes_logging(self):
        with ignore_bytes_warning():
            self.logger.error(b'\xe9')
            self.assertEqual(self.get_output(), utf8(repr(b'\xe9')))

    def test_utf8_logging(self):
        with ignore_bytes_warning():
            self.logger.error('é'.encode('utf8'))
        if issubclass(bytes, basestring_type):
            self.assertEqual(self.get_output(), utf8('é'))
        else:
            self.assertEqual(self.get_output(), utf8(repr(utf8('é'))))

    def test_bytes_exception_logging(self):
        try:
            raise Exception(b'\xe9')
        except Exception:
            self.logger.exception('caught exception')
        output = self.get_output()
        self.assertRegex(output, b'Exception.*\\\\xe9')
        self.assertNotIn(b'\\n', output)

    def test_unicode_logging(self):
        self.logger.error('é')
        self.assertEqual(self.get_output(), utf8('é'))