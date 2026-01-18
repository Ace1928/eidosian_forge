import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
def _CheckGeneratedFiles(self, api_name, api_version):
    prefix = api_name + '_' + api_version
    with test_utils.TempDir() as tmp_dir_path:
        gen_client.main([gen_client.__file__, '--init-file', 'empty', '--infile', GetSampleClientPath(api_name, prefix + '.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'samples.{0}_sample.{0}_{1}'.format(api_name, api_version), 'client'])
        expected_files = set([prefix + '_client.py', prefix + '_messages.py', '__init__.py'])
        self.assertEquals(expected_files, set(os.listdir(tmp_dir_path)))
        if six.PY3:
            return
        for expected_file in expected_files:
            self.AssertDiffEqual(_GetContent(GetSampleClientPath(api_name, prefix, expected_file)), _GetContent(os.path.join(tmp_dir_path, expected_file)))