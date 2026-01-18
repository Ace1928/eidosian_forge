import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def _do_container_action(self, container, action, timeout, force, stateful):
    """
        change the container state by performing the given action
        action may be either stop, start, restart, freeze or unfreeze
        """
    if action not in LXD_API_STATE_ACTIONS:
        raise ValueError('Invalid action specified')
    state = container.state
    data = {'action': action, 'timeout': timeout}
    data = json.dumps(data)
    req = '/{}/containers/{}/state'.format(self.version, container.name)
    response = self.connection.request(req, method='PUT', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=100)
    if not timeout:
        timeout = LXDContainerDriver.default_time_out
    try:
        id = response_dict['metadata']['id']
        req = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, timeout)
        response = self.connection.request(req)
    except BaseHTTPError as err:
        lxd_exception = self._get_lxd_api_exception_for_error(err)
        if lxd_exception.message != 'not found':
            raise lxd_exception
    if state == ContainerState.RUNNING and container.extra['ephemeral'] and (action == 'stop'):
        container = Container(driver=self, name=container.name, id=container.name, state=ContainerState.TERMINATED, image=None, ip_addresses=[], extra=None)
        return container
    return self.get_container(id=container.name)