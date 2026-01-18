import sys
from libcloud.test import unittest
from libcloud.compute.drivers.ikoula import IkoulaNodeDriver
from libcloud.test.compute.test_cloudstack import CloudStackCommonTestCase
class IkoulaNodeDriverTestCase(CloudStackCommonTestCase, unittest.TestCase):
    driver_klass = IkoulaNodeDriver