import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestPassthruListBaremetalDriver(TestBaremetalDriver):

    def setUp(self):
        super(TestPassthruListBaremetalDriver, self).setUp()
        self.baremetal_mock.driver.get_vendor_passthru_methods.return_value = baremetal_fakes.BAREMETAL_DRIVER_PASSTHRU
        self.cmd = baremetal_driver.PassthruListBaremetalDriver(self.app, None)

    def test_baremetal_driver_passthru_list(self):
        arglist = ['fakedrivername']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['fakedrivername']
        self.baremetal_mock.driver.get_vendor_passthru_methods.assert_called_with(*args)
        collist = ('Name', 'Supported HTTP methods', 'Async', 'Description', 'Response is attachment')
        self.assertEqual(collist, tuple(columns))
        datalist = (('lookup', 'POST', 'false', '', 'false'),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_driver_passthru_list_no_arg(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)