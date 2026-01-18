import boto
from boto.codedeploy.exceptions import ApplicationDoesNotExistException
from tests.compat import unittest
class TestCodeDeploy(unittest.TestCase):

    def setUp(self):
        self.codedeploy = boto.connect_codedeploy()

    def test_applications(self):
        application_name = 'my-boto-application'
        self.codedeploy.create_application(application_name=application_name)
        self.addCleanup(self.codedeploy.delete_application, application_name)
        response = self.codedeploy.list_applications()
        self.assertIn(application_name, response['applications'])

    def test_exception(self):
        with self.assertRaises(ApplicationDoesNotExistException):
            self.codedeploy.get_application('some-non-existant-app')