import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def _degen_image(self, image):
    """
        Take in an image object to break down into an ``imageUuid``

        :param image:
        :return:
        """
    image_type = 'docker'
    if image.version is not None:
        return image_type + ':' + image.path + ':' + image.version
    else:
        return image_type + ':' + image.path