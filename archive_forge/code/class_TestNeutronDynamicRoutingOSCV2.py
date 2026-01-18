from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
class TestNeutronDynamicRoutingOSCV2(fakes.TestNeutronClientOSCV2):

    def setUp(self):
        super(TestNeutronDynamicRoutingOSCV2, self).setUp()
        self.neutronclient.find_resource = mock.Mock(side_effect=lambda resource, name_or_id, project_id=None, cmd_resource=None, parent_id=None, fields=None: {'id': name_or_id})
        self.networkclient.find_bgp_speaker = mock.Mock(side_effect=lambda name_or_id, project_id=None, cmd_resource=None, parent_id=None, fields=None, ignore_missing=False: _bgp_speaker.BgpSpeaker(id=name_or_id))
        self.networkclient.find_bgp_peer = mock.Mock(side_effect=lambda name_or_id, project_id=None, cmd_resource=None, parent_id=None, fields=None, ignore_missing=False: _bgp_peer.BgpPeer(id=name_or_id))