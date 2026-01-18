import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def _gen_image(self, imageuuid):
    """
        This function converts a valid Rancher ``imageUuid`` string to a valid
        image object. Only supports docker based images hence `docker:` must
        prefix!!

        Please see the deploy_container() for details on the format.

        :param imageuuid: A valid Rancher image string
            i.e. ``docker:rlister/hastebin:8.0``
        :type imageuuid: ``str``

        :return: Converted ContainerImage object.
        :rtype: :class:`libcloud.container.base.ContainerImage`
        """
    if '/' not in imageuuid:
        image_name_version = imageuuid.partition(':')[2]
    else:
        image_name_version = imageuuid.rpartition('/')[2]
    if ':' in image_name_version:
        version = image_name_version.partition(':')[2]
        id = image_name_version.partition(':')[0]
        name = id
    else:
        version = 'latest'
        id = image_name_version
        name = id
    if version != 'latest':
        path = imageuuid.partition(':')[2].rpartition(':')[0]
    else:
        path = imageuuid.partition(':')[2]
    return ContainerImage(id=id, name=name, path=path, version=version, driver=self.connection.driver, extra={'imageUuid': imageuuid})