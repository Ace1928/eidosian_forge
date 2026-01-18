import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
class ClientGenCliTest(unittest.TestCase):

    def testHelp_NotEnoughArguments(self):
        with self.assertRaisesRegexp(SystemExit, '0'):
            with test_utils.CaptureOutput() as (_, err):
                gen_client.main([gen_client.__file__, '-h'])
                err_output = err.getvalue()
                self.assertIn('usage:', err_output)
                self.assertIn('error: too few arguments', err_output)

    def testGenClient_SimpleDocNoInit(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--init-file', 'none', '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'client'])
            expected_files = set(['dns_v1_client.py', 'dns_v1_messages.py'])
            self.assertEquals(expected_files, set(os.listdir(tmp_dir_path)))

    def testGenClient_SimpleDocEmptyInit(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--init-file', 'empty', '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'client'])
            expected_files = set(['dns_v1_client.py', 'dns_v1_messages.py', '__init__.py'])
            self.assertEquals(expected_files, set(os.listdir(tmp_dir_path)))
            init_file = _GetContent(os.path.join(tmp_dir_path, '__init__.py'))
            self.assertEqual('"""Package marker file."""\n\nfrom __future__ import absolute_import\n\nimport pkgutil\n\n__path__ = pkgutil.extend_path(__path__, __name__)\n', init_file)

    def testGenClient_SimpleDocWithV4(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--apitools_version', '0.4.12', '--root_package', 'google.apis', 'client'])
            self.assertEquals(set(['dns_v1_client.py', 'dns_v1_messages.py', '__init__.py']), set(os.listdir(tmp_dir_path)))

    def testGenClient_SimpleDocWithV5(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--apitools_version', '0.5.0', '--root_package', 'google.apis', 'client'])
            self.assertEquals(set(['dns_v1_client.py', 'dns_v1_messages.py', '__init__.py']), set(os.listdir(tmp_dir_path)))

    def testGenPipPackage_SimpleDoc(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'pip_package'])
            self.assertEquals(set(['apitools', 'setup.py']), set(os.listdir(tmp_dir_path)))

    def testGenProto_SimpleDoc(self):
        with test_utils.TempDir() as tmp_dir_path:
            gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'proto'])
            self.assertEquals(set(['dns_v1_messages.proto', 'dns_v1_services.proto']), set(os.listdir(tmp_dir_path)))