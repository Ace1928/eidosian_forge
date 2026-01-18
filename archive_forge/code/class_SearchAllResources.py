from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SearchAllResources(SearchAllResourcesBeta):
    """Searches all Cloud resources within the specified accessible scope, such as a project, folder or organization."""

    @staticmethod
    def Args(parser):
        AddScopeArgument(parser)
        AddQueryArgument(parser)
        AddAssetTypesArgument(parser)
        AddOrderByArgument(parser)
        AddReadMaskArgument(parser)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        client = client_util.AssetSearchClient(client_util.DEFAULT_API_VERSION)
        return client.SearchAllResources(args)