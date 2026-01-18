from openstack import exceptions
from openstack import resource
from openstack import utils
class BgpSpeaker(resource.Resource):
    resource_key = 'bgp_speaker'
    resources_key = 'bgp_speakers'
    base_path = '/bgp-speakers'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    id = resource.Body('id')
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    ip_version = resource.Body('ip_version')
    advertise_floating_ip_host_routes = resource.Body('advertise_floating_ip_host_routes')
    advertise_tenant_networks = resource.Body('advertise_tenant_networks')
    local_as = resource.Body('local_as')
    networks = resource.Body('networks')

    def _put(self, session, url, body):
        resp = session.put(url, json=body)
        exceptions.raise_from_response(resp)
        return resp

    def add_bgp_peer(self, session, peer_id):
        """Add BGP Peer to a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param peer_id: id of the peer to associate with the speaker.

        :returns: A dictionary as the API Reference describes it.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
        url = utils.urljoin(self.base_path, self.id, 'add_bgp_peer')
        body = {'bgp_peer_id': peer_id}
        resp = self._put(session, url, body)
        return resp.json()

    def remove_bgp_peer(self, session, peer_id):
        """Remove BGP Peer from a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param peer_id: The ID of the peer to disassociate from the speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
        url = utils.urljoin(self.base_path, self.id, 'remove_bgp_peer')
        body = {'bgp_peer_id': peer_id}
        self._put(session, url, body)

    def add_gateway_network(self, session, network_id):
        """Add Network to a BGP Speaker

        :param: session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param network_id: The ID of the network to associate with the speaker

        :returns: A dictionary as the API Reference describes it.
        """
        body = {'network_id': network_id}
        url = utils.urljoin(self.base_path, self.id, 'add_gateway_network')
        resp = session.put(url, json=body)
        return resp.json()

    def remove_gateway_network(self, session, network_id):
        """Delete Network from a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param network_id: The ID of the network to disassociate
               from the speaker
        """
        body = {'network_id': network_id}
        url = utils.urljoin(self.base_path, self.id, 'remove_gateway_network')
        session.put(url, json=body)

    def get_advertised_routes(self, session):
        """List routes advertised by a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: The response as a list of routes (cidr/nexthop pair
                  advertised by the BGP Speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
        url = utils.urljoin(self.base_path, self.id, 'get_advertised_routes')
        resp = session.get(url)
        exceptions.raise_from_response(resp)
        self._body.attributes.update(resp.json())
        return resp.json()

    def get_bgp_dragents(self, session):
        """List Dynamic Routing Agents hosting a specific BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: The response as a list of dragents hosting a specific
                  BGP Speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
        url = utils.urljoin(self.base_path, self.id, 'bgp-dragents')
        resp = session.get(url)
        exceptions.raise_from_response(resp)
        self._body.attributes.update(resp.json())
        return resp.json()

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

    def remove_bgp_speaker_from_dragent(self, session, bgp_agent_id):
        """Delete BGP Speaker from a Dynamic Routing Agent

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param bgp_agent_id: The id of the dynamic routing agent from which
                             remove the speaker.
        """
        url = utils.urljoin('agents', bgp_agent_id, 'bgp-drinstances', self.id)
        session.delete(url)