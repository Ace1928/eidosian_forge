from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def _to_clusters(self, data):
    clusters = []
    for cluster in data['clusters']:
        clusters.append(self._to_cluster(cluster))
    return clusters