from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
class AssetSearchClient(object):
    """Client for search assets."""
    _DEFAULT_PAGE_SIZE = 20

    def __init__(self, api_version):
        self.message_module = GetMessages(api_version)
        self.api_version = api_version
        if api_version == V1P1BETA1_API_VERSION:
            self.resource_service = GetClient(api_version).resources
            self.search_all_resources_method = 'SearchAll'
            self.search_all_resources_request = self.message_module.CloudassetResourcesSearchAllRequest
            self.policy_service = GetClient(api_version).iamPolicies
            self.search_all_iam_policies_method = 'SearchAll'
            self.search_all_iam_policies_request = self.message_module.CloudassetIamPoliciesSearchAllRequest
        else:
            self.resource_service = GetClient(api_version).v1
            self.search_all_resources_method = 'SearchAllResources'
            self.search_all_resources_request = self.message_module.CloudassetSearchAllResourcesRequest
            self.policy_service = GetClient(api_version).v1
            self.search_all_iam_policies_method = 'SearchAllIamPolicies'
            self.search_all_iam_policies_request = self.message_module.CloudassetSearchAllIamPoliciesRequest

    def SearchAllResources(self, args):
        """Calls SearchAllResources method."""
        if self.api_version == V1P1BETA1_API_VERSION:
            optional_extra_args = {}
        else:
            optional_extra_args = {'readMask': args.read_mask}
        request = self.search_all_resources_request(scope=asset_utils.GetDefaultScopeIfEmpty(args), query=args.query, assetTypes=args.asset_types, orderBy=args.order_by, **optional_extra_args)
        return list_pager.YieldFromList(self.resource_service, request, method=self.search_all_resources_method, field='results', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')

    def SearchAllIamPolicies(self, args):
        """Calls SearchAllIamPolicies method."""
        if self.api_version == V1P1BETA1_API_VERSION:
            request = self.search_all_iam_policies_request(scope=asset_utils.GetDefaultScopeIfEmpty(args), query=args.query)
        else:
            request = self.search_all_iam_policies_request(scope=asset_utils.GetDefaultScopeIfEmpty(args), query=args.query, assetTypes=args.asset_types, orderBy=args.order_by)
        return list_pager.YieldFromList(self.policy_service, request, method=self.search_all_iam_policies_method, field='results', batch_size=args.page_size or self._DEFAULT_PAGE_SIZE, batch_size_attribute='pageSize', current_token_attribute='pageToken', next_token_attribute='nextPageToken')