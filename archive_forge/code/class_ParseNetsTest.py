import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
class ParseNetsTest(test_utils.BaseTestCase):

    def test_no_nets(self):
        nets = []
        result = utils.parse_nets(nets)
        self.assertEqual([], result)

    def test_nets_with_network(self):
        nets = [' network = 1234567 , v4-fixed-ip = 172.17.0.3 ']
        result = utils.parse_nets(nets)
        self.assertEqual([{'network': '1234567', 'v4-fixed-ip': '172.17.0.3'}], result)

    def test_nets_with_port(self):
        nets = ['port=1234567, v6-fixed-ip=2001:db8::2']
        result = utils.parse_nets(nets)
        self.assertEqual([{'port': '1234567', 'v6-fixed-ip': '2001:db8::2'}], result)

    def test_nets_with_only_ip(self):
        nets = ['v4-fixed-ip = 172.17.0.3']
        self.assertRaises(exc.CommandError, utils.parse_nets, nets)

    def test_nets_with_both_network_port(self):
        nets = ['port=1234567, network=2345678, v4-fixed-ip=172.17.0.3']
        self.assertRaises(exc.CommandError, utils.parse_nets, nets)

    def test_nets_with_invalid_ip(self):
        nets = ['network=1234567, v4-fixed-ip=23.555.567,789']
        self.assertRaises(exc.CommandError, utils.parse_nets, nets)