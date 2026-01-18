import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
class AgentInteractor(httpbakery.Interactor, httpbakery.LegacyInteractor):
    """ Interactor that performs interaction using the agent login protocol.
    """

    def __init__(self, auth_info):
        self._auth_info = auth_info

    def kind(self):
        """Implement Interactor.kind by returning the agent kind"""
        return 'agent'

    def interact(self, client, location, interaction_required_err):
        """Implement Interactor.interact by obtaining obtaining
        a macaroon from the discharger, discharging it with the
        local private key using the discharged macaroon as
        a discharge token"""
        p = interaction_required_err.interaction_method('agent', InteractionInfo)
        if p.login_url is None or p.login_url == '':
            raise httpbakery.InteractionError('no login-url field found in agent interaction method')
        agent = self._find_agent(location)
        if not location.endswith('/'):
            location += '/'
        login_url = urljoin(location, p.login_url)
        resp = requests.get(login_url, params={'username': agent.username, 'public-key': str(self._auth_info.key.public_key)}, auth=client.auth())
        if resp.status_code != 200:
            raise httpbakery.InteractionError('cannot acquire agent macaroon: {} {}'.format(resp.status_code, resp.text))
        m = resp.json().get('macaroon')
        if m is None:
            raise httpbakery.InteractionError('no macaroon in response')
        m = bakery.Macaroon.from_dict(m)
        ms = bakery.discharge_all(m, None, self._auth_info.key)
        b = bytearray()
        for m in ms:
            b.extend(utils.b64decode(m.serialize()))
        return httpbakery.DischargeToken(kind='agent', value=bytes(b))

    def _find_agent(self, location):
        """ Finds an appropriate agent entry for the given location.
        :return Agent
        """
        for a in self._auth_info.agents:
            if a.url.rstrip('/') == location.rstrip('/'):
                return a
        raise httpbakery.InteractionMethodNotFound('cannot find username for discharge location {}'.format(location))

    def legacy_interact(self, client, location, visit_url):
        """Implement LegacyInteractor.legacy_interact by obtaining
        the discharge macaroon using the client's private key
        """
        agent = self._find_agent(location)
        client = copy.copy(client)
        client.key = self._auth_info.key
        resp = client.request(method='POST', url=visit_url, json={'username': agent.username, 'public_key': str(self._auth_info.key.public_key)})
        if resp.status_code != 200:
            raise httpbakery.InteractionError('cannot acquire agent macaroon from {}: {} (response body: {!r})'.format(visit_url, resp.status_code, resp.text))
        if not resp.json().get('agent_login', False):
            raise httpbakery.InteractionError('agent login failed')