import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneUnset(TestAvailabilityzone):
    PARAMETERS = ('description',)

    def setUp(self):
        super().setUp()
        self.cmd = availabilityzone.UnsetAvailabilityzone(self.app, None)

    def test_hm_unset_description(self):
        self._test_availabilityzone_unset_param('description')

    def _test_availabilityzone_unset_param(self, param):
        self.api_mock.availabilityzone_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._availabilityzone.name, '--%s' % arg_param]
        ref_body = {'availability_zone': {param: None}}
        verifylist = [('availabilityzone', self._availabilityzone.name)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        print(verifylist)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzone_set.assert_called_once_with(self._availabilityzone.name, json=ref_body)

    def test_availabilityzone_unset_all(self):
        self.api_mock.availabilityzone_set.reset_mock()
        ref_body = {'availability_zone': {x: None for x in self.PARAMETERS}}
        arglist = [self._availabilityzone.name]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('availabilityzone', self._availabilityzone.name)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzone_set.assert_called_once_with(self._availabilityzone.name, json=ref_body)

    def test_availabilityzone_unset_none(self):
        self.api_mock.availabilityzone_set.reset_mock()
        arglist = [self._availabilityzone.name]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('availabilityzone', self._availabilityzone.name)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzone_set.assert_not_called()