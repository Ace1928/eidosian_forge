from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAuxiliaryFunctions(tests_utils.TestCase):

    def test__get_ranges(self):
        input_reference = [([1, 2, 3, 4, 5, 6, 7], ['1-7']), ([1, 2, 5, 4, 3, 6, 7], ['1-7']), ([1, 2, 4, 3, 7, 6], ['1-4', '6-7']), ([1, 2, 4, 3, '13', 12, '7', '6'], ['1-4', '6-7', '12-13'])]
        for input, reference in input_reference:
            self.assertEqual(reference, list(network_segment_range._get_ranges(input)))