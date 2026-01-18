import sys
from libcloud.test import unittest
from libcloud.test.compute.test_cloudstack import CloudStackCommonTestCase
from libcloud.compute.drivers.auroracompute import AuroraComputeRegion, AuroraComputeNodeDriver
class AuroraComputeNodeDriverTestCase(CloudStackCommonTestCase, unittest.TestCase):
    driver_klass = AuroraComputeNodeDriver

    def test_api_host(self):
        driver = self.driver_klass('invalid', 'invalid')
        self.assertEqual(driver.host, 'api.auroracompute.eu')

    def test_without_region(self):
        driver = self.driver_klass('invalid', 'invalid')
        self.assertEqual(driver.path, '/ams')

    def test_with_ams_region(self):
        driver = self.driver_klass('invalid', 'invalid', region=AuroraComputeRegion.AMS)
        self.assertEqual(driver.path, '/ams')

    def test_with_miami_region(self):
        driver = self.driver_klass('invalid', 'invalid', region=AuroraComputeRegion.MIA)
        self.assertEqual(driver.path, '/mia')

    def test_with_tokyo_region(self):
        driver = self.driver_klass('invalid', 'invalid', region=AuroraComputeRegion.TYO)
        self.assertEqual(driver.path, '/tyo')