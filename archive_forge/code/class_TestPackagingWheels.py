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
class TestPackagingWheels(base.BaseTestCase):

    def setUp(self):
        super(TestPackagingWheels, self).setUp()
        self.useFixture(TestRepo(self.package_dir))
        self.run_setup('bdist_wheel', allow_fail=False)
        dist_dir = os.path.join(self.package_dir, 'dist')
        relative_wheel_filename = os.listdir(dist_dir)[0]
        absolute_wheel_filename = os.path.join(dist_dir, relative_wheel_filename)
        wheel_file = wheelfile.WheelFile(absolute_wheel_filename)
        wheel_name = wheel_file.parsed_filename.group('namever')
        self.extracted_wheel_dir = os.path.join(dist_dir, wheel_name)
        wheel_file.extractall(self.extracted_wheel_dir)
        wheel_file.close()

    def test_metadata_directory_has_pbr_json(self):
        pbr_json = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage-0.0.dist-info/pbr.json')
        self.assertTrue(os.path.exists(pbr_json))

    def test_data_directory_has_wsgi_scripts(self):
        scripts_dir = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage-0.0.data/scripts')
        self.assertTrue(os.path.exists(scripts_dir))
        scripts = os.listdir(scripts_dir)
        self.assertIn('pbr_test_wsgi', scripts)
        self.assertIn('pbr_test_wsgi_with_class', scripts)
        self.assertNotIn('pbr_test_cmd', scripts)
        self.assertNotIn('pbr_test_cmd_with_class', scripts)

    def test_generates_c_extensions(self):
        built_package_dir = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage')
        static_object_filename = 'testext.so'
        soabi = get_soabi()
        if soabi:
            static_object_filename = 'testext.{0}.so'.format(soabi)
        static_object_path = os.path.join(built_package_dir, static_object_filename)
        self.assertTrue(os.path.exists(built_package_dir))
        self.assertTrue(os.path.exists(static_object_path))