from openstack import exceptions
from openstack import resource
from openstack import utils
def add_bgp_speaker_to_dragent(self, session, bgp_agent_id):
    """Add BGP Speaker to a Dynamic Routing Agent

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param bgp_agent_id: The id of the dynamic routing agent to which
                             add the speaker.
        """
    body = {'bgp_speaker_id': self.id}
    url = utils.urljoin('agents', bgp_agent_id, 'bgp-drinstances')
    session.post(url, json=body)