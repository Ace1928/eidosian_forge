from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestDeleteBgpSpeaker(fakes.TestNeutronDynamicRoutingOSCV2):
    _bgp_speaker = fakes.FakeBgpSpeaker.create_one_bgp_speaker()

    def setUp(self):
        super(TestDeleteBgpSpeaker, self).setUp()
        self.networkclient.delete_bgp_speaker = mock.Mock(return_value=None)
        self.cmd = bgp_speaker.DeleteBgpSpeaker(self.app, self.namespace)

    def test_delete_bgp_speaker(self):
        arglist = [self._bgp_speaker['name']]
        verifylist = [('bgp_speaker', self._bgp_speaker['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.networkclient.delete_bgp_speaker.assert_called_once_with(self._bgp_speaker['name'])
        self.assertIsNone(result)