from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestListBgpSpeaker(fakes.TestNeutronDynamicRoutingOSCV2):
    _bgp_speakers = fakes.FakeBgpSpeaker.create_bgp_speakers()
    columns = ('ID', 'Name', 'Local AS', 'IP Version')
    data = []
    for _bgp_speaker in _bgp_speakers:
        data.append((_bgp_speaker['id'], _bgp_speaker['name'], _bgp_speaker['local_as'], _bgp_speaker['ip_version']))

    def setUp(self):
        super(TestListBgpSpeaker, self).setUp()
        self.networkclient.bgp_speakers = mock.Mock(return_value=self._bgp_speakers)
        self.cmd = bgp_speaker.ListBgpSpeaker(self.app, self.namespace)

    def test_bgp_speaker_list(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgp_speakers.assert_called_once_with(retrieve_all=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))