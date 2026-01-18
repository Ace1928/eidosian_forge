import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import AllTestsBase
import logging
class RoundTripJSONTests(RoundTripTestCase, AllTestsBase):
    FORMAT = 'json'