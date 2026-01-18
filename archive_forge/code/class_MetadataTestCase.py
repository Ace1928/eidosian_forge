import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
class MetadataTestCase(support.TempdirManager, support.EnvironGuard, unittest.TestCase):

    def setUp(self):
        super(MetadataTestCase, self).setUp()
        self.argv = (sys.argv, sys.argv[:])

    def tearDown(self):
        sys.argv = self.argv[0]
        sys.argv[:] = self.argv[1]
        super(MetadataTestCase, self).tearDown()

    def format_metadata(self, dist):
        sio = io.StringIO()
        dist.metadata.write_pkg_file(sio)
        return sio.getvalue()

    def test_simple_metadata(self):
        attrs = {'name': 'package', 'version': '1.0'}
        dist = Distribution(attrs)
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.0', meta)
        self.assertNotIn('provides:', meta.lower())
        self.assertNotIn('requires:', meta.lower())
        self.assertNotIn('obsoletes:', meta.lower())

    def test_provides(self):
        attrs = {'name': 'package', 'version': '1.0', 'provides': ['package', 'package.sub']}
        dist = Distribution(attrs)
        self.assertEqual(dist.metadata.get_provides(), ['package', 'package.sub'])
        self.assertEqual(dist.get_provides(), ['package', 'package.sub'])
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.1', meta)
        self.assertNotIn('requires:', meta.lower())
        self.assertNotIn('obsoletes:', meta.lower())

    def test_provides_illegal(self):
        self.assertRaises(ValueError, Distribution, {'name': 'package', 'version': '1.0', 'provides': ['my.pkg (splat)']})

    def test_requires(self):
        attrs = {'name': 'package', 'version': '1.0', 'requires': ['other', 'another (==1.0)']}
        dist = Distribution(attrs)
        self.assertEqual(dist.metadata.get_requires(), ['other', 'another (==1.0)'])
        self.assertEqual(dist.get_requires(), ['other', 'another (==1.0)'])
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.1', meta)
        self.assertNotIn('provides:', meta.lower())
        self.assertIn('Requires: other', meta)
        self.assertIn('Requires: another (==1.0)', meta)
        self.assertNotIn('obsoletes:', meta.lower())

    def test_requires_illegal(self):
        self.assertRaises(ValueError, Distribution, {'name': 'package', 'version': '1.0', 'requires': ['my.pkg (splat)']})

    def test_requires_to_list(self):
        attrs = {'name': 'package', 'requires': iter(['other'])}
        dist = Distribution(attrs)
        self.assertIsInstance(dist.metadata.requires, list)

    def test_obsoletes(self):
        attrs = {'name': 'package', 'version': '1.0', 'obsoletes': ['other', 'another (<1.0)']}
        dist = Distribution(attrs)
        self.assertEqual(dist.metadata.get_obsoletes(), ['other', 'another (<1.0)'])
        self.assertEqual(dist.get_obsoletes(), ['other', 'another (<1.0)'])
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.1', meta)
        self.assertNotIn('provides:', meta.lower())
        self.assertNotIn('requires:', meta.lower())
        self.assertIn('Obsoletes: other', meta)
        self.assertIn('Obsoletes: another (<1.0)', meta)

    def test_obsoletes_illegal(self):
        self.assertRaises(ValueError, Distribution, {'name': 'package', 'version': '1.0', 'obsoletes': ['my.pkg (splat)']})

    def test_obsoletes_to_list(self):
        attrs = {'name': 'package', 'obsoletes': iter(['other'])}
        dist = Distribution(attrs)
        self.assertIsInstance(dist.metadata.obsoletes, list)

    def test_classifier(self):
        attrs = {'name': 'Boa', 'version': '3.0', 'classifiers': ['Programming Language :: Python :: 3']}
        dist = Distribution(attrs)
        self.assertEqual(dist.get_classifiers(), ['Programming Language :: Python :: 3'])
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.1', meta)

    def test_classifier_invalid_type(self):
        attrs = {'name': 'Boa', 'version': '3.0', 'classifiers': ('Programming Language :: Python :: 3',)}
        with captured_stderr() as error:
            d = Distribution(attrs)
        self.assertIn('should be a list', error.getvalue())
        self.assertIsInstance(d.metadata.classifiers, list)
        self.assertEqual(d.metadata.classifiers, list(attrs['classifiers']))

    def test_keywords(self):
        attrs = {'name': 'Monty', 'version': '1.0', 'keywords': ['spam', 'eggs', 'life of brian']}
        dist = Distribution(attrs)
        self.assertEqual(dist.get_keywords(), ['spam', 'eggs', 'life of brian'])

    def test_keywords_invalid_type(self):
        attrs = {'name': 'Monty', 'version': '1.0', 'keywords': ('spam', 'eggs', 'life of brian')}
        with captured_stderr() as error:
            d = Distribution(attrs)
        self.assertIn('should be a list', error.getvalue())
        self.assertIsInstance(d.metadata.keywords, list)
        self.assertEqual(d.metadata.keywords, list(attrs['keywords']))

    def test_platforms(self):
        attrs = {'name': 'Monty', 'version': '1.0', 'platforms': ['GNU/Linux', 'Some Evil Platform']}
        dist = Distribution(attrs)
        self.assertEqual(dist.get_platforms(), ['GNU/Linux', 'Some Evil Platform'])

    def test_platforms_invalid_types(self):
        attrs = {'name': 'Monty', 'version': '1.0', 'platforms': ('GNU/Linux', 'Some Evil Platform')}
        with captured_stderr() as error:
            d = Distribution(attrs)
        self.assertIn('should be a list', error.getvalue())
        self.assertIsInstance(d.metadata.platforms, list)
        self.assertEqual(d.metadata.platforms, list(attrs['platforms']))

    def test_download_url(self):
        attrs = {'name': 'Boa', 'version': '3.0', 'download_url': 'http://example.org/boa'}
        dist = Distribution(attrs)
        meta = self.format_metadata(dist)
        self.assertIn('Metadata-Version: 1.1', meta)

    def test_long_description(self):
        long_desc = textwrap.dedent('        example::\n              We start here\n            and continue here\n          and end here.')
        attrs = {'name': 'package', 'version': '1.0', 'long_description': long_desc}
        dist = Distribution(attrs)
        meta = self.format_metadata(dist)
        meta = meta.replace('\n' + 8 * ' ', '\n')
        self.assertIn(long_desc, meta)

    def test_custom_pydistutils(self):
        if os.name == 'posix':
            user_filename = '.pydistutils.cfg'
        else:
            user_filename = 'pydistutils.cfg'
        temp_dir = self.mkdtemp()
        user_filename = os.path.join(temp_dir, user_filename)
        f = open(user_filename, 'w')
        try:
            f.write('.')
        finally:
            f.close()
        try:
            dist = Distribution()
            if sys.platform in ('linux', 'darwin'):
                os.environ['HOME'] = temp_dir
                files = dist.find_config_files()
                self.assertIn(user_filename, files)
            if sys.platform == 'win32':
                os.environ['USERPROFILE'] = temp_dir
                files = dist.find_config_files()
                self.assertIn(user_filename, files, '%r not found in %r' % (user_filename, files))
        finally:
            os.remove(user_filename)

    def test_fix_help_options(self):
        help_tuples = [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
        fancy_options = fix_help_options(help_tuples)
        self.assertEqual(fancy_options[0], ('a', 'b', 'c'))
        self.assertEqual(fancy_options[1], (1, 2, 3))

    def test_show_help(self):
        self.addCleanup(log.set_threshold, log._global_log.threshold)
        dist = Distribution()
        sys.argv = []
        dist.help = 1
        dist.script_name = 'setup.py'
        with captured_stdout() as s:
            dist.parse_command_line()
        output = [line for line in s.getvalue().split('\n') if line.strip() != '']
        self.assertTrue(output)

    def test_read_metadata(self):
        attrs = {'name': 'package', 'version': '1.0', 'long_description': 'desc', 'description': 'xxx', 'download_url': 'http://example.com', 'keywords': ['one', 'two'], 'requires': ['foo']}
        dist = Distribution(attrs)
        metadata = dist.metadata
        PKG_INFO = io.StringIO()
        metadata.write_pkg_file(PKG_INFO)
        PKG_INFO.seek(0)
        metadata.read_pkg_file(PKG_INFO)
        self.assertEqual(metadata.name, 'package')
        self.assertEqual(metadata.version, '1.0')
        self.assertEqual(metadata.description, 'xxx')
        self.assertEqual(metadata.download_url, 'http://example.com')
        self.assertEqual(metadata.keywords, ['one', 'two'])
        self.assertEqual(metadata.platforms, ['UNKNOWN'])
        self.assertEqual(metadata.obsoletes, None)
        self.assertEqual(metadata.requires, ['foo'])