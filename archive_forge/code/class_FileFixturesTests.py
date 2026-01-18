import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
class FileFixturesTests(unittest.TestCase):

    def test_success(self):
        f = ComputeFileFixtures('meta')
        self.assertEqual('Hello, World!', f.load('helloworld.txt'))

    def test_failure(self):
        f = ComputeFileFixtures('meta')
        self.assertRaises(IOError, f.load, 'nil')

    def test_unicode(self):
        f = ComputeFileFixtures('meta')
        self.assertEqual('Ś', f.load('unicode.txt'))