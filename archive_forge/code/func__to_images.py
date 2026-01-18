from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def _to_images(self, data, host, repository_name):
    images = []
    for image in data:
        images.append(self._to_image(image, host, repository_name))
    return images