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
class TestRequirementParsing(base.BaseTestCase):

    def test_requirement_parsing(self):
        pkgs = {'test_reqparse': {'requirements.txt': textwrap.dedent("                        bar\n                        quux<1.0; python_version=='2.6'\n                        requests-aws>=0.1.4    # BSD License (3 clause)\n                        Routes>=1.12.3,!=2.0,!=2.1;python_version=='2.7'\n                        requests-kerberos>=0.6;python_version=='2.7' # MIT\n                    "), 'setup.cfg': textwrap.dedent("                        [metadata]\n                        name = test_reqparse\n\n                        [extras]\n                        test =\n                            foo\n                            baz>3.2 :python_version=='2.7' # MIT\n                            bar>3.3 :python_version=='2.7' # MIT # Apache\n                    ")}}
        pkg_dirs = self.useFixture(CreatePackages(pkgs)).package_dirs
        pkg_dir = pkg_dirs['test_reqparse']
        expected_requirements = {None: ['bar', 'requests-aws>=0.1.4'], ":(python_version=='2.6')": ['quux<1.0'], ":(python_version=='2.7')": ['Routes!=2.0,!=2.1,>=1.12.3', 'requests-kerberos>=0.6'], 'test': ['foo'], "test:(python_version=='2.7')": ['baz>3.2', 'bar>3.3']}
        venv = self.useFixture(Venv('reqParse'))
        bin_python = venv.python
        self._run_cmd(bin_python, ('setup.py', 'bdist_wheel'), allow_fail=False, cwd=pkg_dir)
        egg_info = os.path.join(pkg_dir, 'test_reqparse.egg-info')
        requires_txt = os.path.join(egg_info, 'requires.txt')
        with open(requires_txt, 'rt') as requires:
            generated_requirements = dict(pkg_resources.split_sections(requires))
        for section, expected in expected_requirements.items():
            exp_parsed = [pkg_resources.Requirement.parse(s) for s in expected]
            gen_parsed = [pkg_resources.Requirement.parse(s) for s in generated_requirements[section]]
            self.assertEqual(exp_parsed, gen_parsed)