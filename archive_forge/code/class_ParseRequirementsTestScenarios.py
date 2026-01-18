import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
class ParseRequirementsTestScenarios(base.BaseTestCase):
    versioned_scenarios = [('non-versioned', {'versioned': False, 'expected': ['bar']}), ('versioned', {'versioned': True, 'expected': ['bar>=1.2.3']})]
    subdirectory_scenarios = [('non-subdirectory', {'has_subdirectory': False}), ('has-subdirectory', {'has_subdirectory': True})]
    scenarios = [('normal', {'url': 'foo\nbar', 'expected': ['foo', 'bar']}), ('normal_with_comments', {'url': '# this is a comment\nfoo\n# and another one\nbar', 'expected': ['foo', 'bar']}), ('removes_index_lines', {'url': '-f foobar', 'expected': []})]
    scenarios = scenarios + testscenarios.multiply_scenarios([('ssh_egg_url', {'url': 'git+ssh://foo.com/zipball#egg=bar'}), ('git_https_egg_url', {'url': 'git+https://foo.com/zipball#egg=bar'}), ('http_egg_url', {'url': 'https://foo.com/zipball#egg=bar'})], versioned_scenarios, subdirectory_scenarios)
    scenarios = scenarios + testscenarios.multiply_scenarios([('git_egg_url', {'url': 'git://foo.com/zipball#egg=bar', 'name': 'bar'})], [('non-editable', {'editable': False}), ('editable', {'editable': True})], versioned_scenarios, subdirectory_scenarios)

    def test_parse_requirements(self):
        tmp_file = tempfile.NamedTemporaryFile()
        req_string = self.url
        if hasattr(self, 'editable') and self.editable:
            req_string = '-e %s' % req_string
        if hasattr(self, 'versioned') and self.versioned:
            req_string = '%s-1.2.3' % req_string
        if hasattr(self, 'has_subdirectory') and self.has_subdirectory:
            req_string = '%s&subdirectory=baz' % req_string
        with open(tmp_file.name, 'w') as fh:
            fh.write(req_string)
        self.assertEqual(self.expected, packaging.parse_requirements([tmp_file.name]))