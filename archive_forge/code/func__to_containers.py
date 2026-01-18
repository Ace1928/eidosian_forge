from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def _to_containers(self, data, task_definition_arn):
    clusters = []
    for cluster in data['containers']:
        clusters.append(self._to_container(cluster, task_definition_arn))
    return clusters