from __future__ import absolute_import
import re
import gslib.tests.testcase as testcase
from gslib import exception
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
class TestAutoclassUnit(testcase.GsUtilUnitTestCase):

    def test_set_too_few_arguments_fails(self):
        with self.assertRaisesRegex(exception.CommandException, 'command requires at least'):
            self.RunCommand('autoclass', ['set'])

    def test_get_too_few_arguments_fails(self):
        with self.assertRaisesRegex(exception.CommandException, 'command requires at least'):
            self.RunCommand('autoclass', ['get'])

    def test_no_subcommand_fails(self):
        with self.assertRaisesRegex(exception.CommandException, 'command requires at least'):
            self.RunCommand('autoclass', [])

    def test_invalid_subcommand_fails(self):
        with self.assertRaisesRegex(exception.CommandException, 'Invalid subcommand'):
            self.RunCommand('autoclass', ['fakecommand', 'test'])

    def test_gets_multiple_buckets_with_wildcard(self):
        bucket_uri1 = self.CreateBucket(bucket_name='bucket1')
        bucket_uri2 = self.CreateBucket(bucket_name='bucket2')
        stdout = self.RunCommand('autoclass', ['get', 'gs://bucket*'], return_stdout=True)
        self.assertIn(bucket_uri1.bucket_name, stdout)
        self.assertIn(bucket_uri2.bucket_name, stdout)