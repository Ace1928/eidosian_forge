import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def _to_container(self, data):
    """
        Convert container in proper Container instance object
        ** Updating is NOT supported!!

        :param data: API data about container i.e. result.object
        :return: Proper Container object:
         see http://libcloud.readthedocs.io/en/latest/container/api.html

        """
    rancher_state = data['state']
    terminate_condition = ['removed', 'purged']
    if 'running' in rancher_state:
        state = ContainerState.RUNNING
    elif 'stopped' in rancher_state:
        state = ContainerState.STOPPED
    elif 'restarting' in rancher_state:
        state = ContainerState.REBOOTING
    elif 'error' in rancher_state:
        state = ContainerState.ERROR
    elif any((x in rancher_state for x in terminate_condition)):
        state = ContainerState.TERMINATED
    elif data['transitioning'] == 'yes':
        state = ContainerState.PENDING
    else:
        state = ContainerState.UNKNOWN
    extra = data
    return Container(id=data['id'], name=data['name'], image=self._gen_image(data['imageUuid']), ip_addresses=[data['primaryIpAddress']], state=state, driver=self.connection.driver, extra=extra)