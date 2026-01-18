import os
import subprocess
import sys
from distutils import version
import breezy
from .. import tests
class TestSetup(tests.TestCaseInTempDir):

    def test_build_and_install(self):
        """ test cmd `python setup.py build`

        This tests that the build process and man generator run correctly.
        It also can catch new subdirectories that weren't added to setup.py.
        """
        self.source_dir = os.path.dirname(os.path.dirname(breezy.__file__))
        if not os.path.isfile(os.path.join(self.source_dir, 'setup.py')):
            self.skipTest('There is no setup.py file adjacent to the breezy directory')
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            self.knownFailure('rustc can not be found in the GitHub actions environment')
        try:
            import distutils.sysconfig
            makefile_path = distutils.sysconfig.get_makefile_filename()
            if not os.path.exists(makefile_path):
                self.skipTest('You must have the python Makefile installed to run this test. Usually this can be found by installing "python-dev"')
        except ImportError:
            self.skipTest('You must have distutils installed to run this test. Usually this can be found by installing "python-dev"')
        self.log('test_build running from %s' % self.source_dir)
        build_dir = os.path.join(self.test_dir, 'build')
        install_dir = os.path.join(self.test_dir, 'install')
        self.run_setup(['build', '-b', build_dir, 'install', '--root', install_dir])
        self.assertPathExists(install_dir)
        self.run_setup(['clean', '-b', build_dir])

    def run_setup(self, args):
        args = [sys.executable, './setup.py'] + args
        self.log('source base directory: %s', self.source_dir)
        self.log('args: %r', args)
        env = dict(os.environ)
        env['PYTHONPATH'] = ':'.join(sys.path)
        p = subprocess.Popen(args, cwd=self.source_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = p.communicate()
        self.log('stdout: %r', stdout)
        self.log('stderr: %r', stderr)
        self.assertEqual(0, p.returncode, 'invocation of %r failed' % args)