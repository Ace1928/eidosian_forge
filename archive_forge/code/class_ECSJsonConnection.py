from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
class ECSJsonConnection(SignedAWSConnection):
    version = ECS_VERSION
    host = ECS_HOST
    responseCls = AWSJsonResponse
    service_name = 'ecs'