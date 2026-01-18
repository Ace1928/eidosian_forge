from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestSetBgpSpeaker(fakes.TestNeutronDynamicRoutingOSCV2):
    _one_bgp_speaker = fakes.FakeBgpSpeaker.create_one_bgp_speaker()
    _bgp_speaker_name = _one_bgp_speaker['name']

    def setUp(self):
        super(TestSetBgpSpeaker, self).setUp()
        self.networkclient.update_bgp_speaker = mock.Mock(return_value=None)
        self.cmd = bgp_speaker.SetBgpSpeaker(self.app, self.namespace)

    def test_set_bgp_speaker(self):
        arglist = [self._bgp_speaker_name, '--name', 'noob']
        verifylist = [('bgp_speaker', self._bgp_speaker_name), ('name', 'noob')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'noob'}
        self.networkclient.update_bgp_speaker.assert_called_once_with(self._bgp_speaker_name, **attrs)
        self.assertIsNone(result)