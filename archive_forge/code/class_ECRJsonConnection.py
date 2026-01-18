from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
class ECRJsonConnection(SignedAWSConnection):
    version = ECR_VERSION
    host = ECR_HOST
    responseCls = AWSJsonResponse
    service_name = 'ecr'