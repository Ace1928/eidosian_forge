from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
class FakeBgpSpeaker(object):
    """Fake one or more bgp speakers."""

    @staticmethod
    def create_one_bgp_speaker(attrs=None):
        attrs = attrs or {}
        bgp_speaker_attrs = {'peers': [], 'local_as': 200, 'advertise_tenant_networks': True, 'networks': [], 'ip_version': 4, 'advertise_floating_ip_host_routes': True, 'id': uuid.uuid4().hex, 'name': 'bgp-speaker-' + uuid.uuid4().hex, 'tenant_id': uuid.uuid4().hex}
        bgp_speaker_attrs.update(attrs)
        ret_bgp_speaker = _bgp_speaker.BgpSpeaker(**bgp_speaker_attrs)
        return ret_bgp_speaker

    @staticmethod
    def create_bgp_speakers(attrs=None, count=1):
        """Create multiple fake bgp speakers.

        """
        bgp_speakers = []
        for i in range(count):
            bgp_speaker = FakeBgpSpeaker.create_one_bgp_speaker(attrs)
            bgp_speakers.append(bgp_speaker)
        return bgp_speakers