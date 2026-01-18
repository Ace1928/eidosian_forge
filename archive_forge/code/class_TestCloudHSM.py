import boto
from tests.compat import unittest
from boto.cloudhsm.exceptions import InvalidRequestException
class TestCloudHSM(unittest.TestCase):

    def setUp(self):
        self.cloudhsm = boto.connect_cloudhsm()

    def test_hapgs(self):
        label = 'my-hapg'
        response = self.cloudhsm.create_hapg(label=label)
        hapg_arn = response['HapgArn']
        self.addCleanup(self.cloudhsm.delete_hapg, hapg_arn)
        response = self.cloudhsm.list_hapgs()
        self.assertIn(hapg_arn, response['HapgList'])

    def test_validation_exception(self):
        invalid_arn = 'arn:aws:cloudhsm:us-east-1:123456789012:hapg-55214b8d'
        with self.assertRaises(InvalidRequestException):
            self.cloudhsm.describe_hapg(invalid_arn)