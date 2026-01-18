from apitools.base.py import list_pager
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetUriFromResourceFunc(api_version):

    def UriFunc(cluster, **kwargs):
        kwargs['api_version'] = api_version
        kwargs['collection'] = 'edgecontainer.projects.locations.clusters'
        return resources.REGISTRY.Parse(cluster.name, **kwargs).SelfLink()
    return UriFunc