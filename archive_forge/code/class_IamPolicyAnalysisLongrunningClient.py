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
class IamPolicyAnalysisLongrunningClient(object):
    """Client for analyze IAM policy asynchronously."""

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).v1

    def Analyze(self, scope, args):
        """Analyze IAM Policy asynchronously."""
        analysis_query = self.message_module.IamPolicyAnalysisQuery()
        analysis_query.scope = scope
        if args.IsSpecified('full_resource_name'):
            analysis_query.resourceSelector = self.message_module.ResourceSelector(fullResourceName=args.full_resource_name)
        if args.IsSpecified('identity'):
            analysis_query.identitySelector = self.message_module.IdentitySelector(identity=args.identity)
        if args.IsSpecified('roles') or args.IsSpecified('permissions'):
            analysis_query.accessSelector = self.message_module.AccessSelector()
            if args.IsSpecified('roles'):
                analysis_query.accessSelector.roles.extend(args.roles)
            if args.IsSpecified('permissions'):
                analysis_query.accessSelector.permissions.extend(args.permissions)
        output_config = None
        if args.gcs_output_path:
            output_config = self.message_module.IamPolicyAnalysisOutputConfig(gcsDestination=self.message_module.GoogleCloudAssetV1GcsDestination(uri=args.gcs_output_path))
        else:
            output_config = self.message_module.IamPolicyAnalysisOutputConfig(bigqueryDestination=self.message_module.GoogleCloudAssetV1BigQueryDestination(dataset=args.bigquery_dataset, tablePrefix=args.bigquery_table_prefix))
            if args.IsSpecified('bigquery_partition_key'):
                output_config.bigqueryDestination.partitionKey = getattr(self.message_module.GoogleCloudAssetV1BigQueryDestination.PartitionKeyValueValuesEnum, args.bigquery_partition_key)
            if args.IsSpecified('bigquery_write_disposition'):
                output_config.bigqueryDestination.writeDisposition = args.bigquery_write_disposition
        options = self.message_module.Options()
        if args.expand_groups:
            options.expandGroups = args.expand_groups
        if args.expand_resources:
            options.expandResources = args.expand_resources
        if args.expand_roles:
            options.expandRoles = args.expand_roles
        if args.output_resource_edges:
            options.outputResourceEdges = args.output_resource_edges
        if args.output_group_edges:
            options.outputGroupEdges = args.output_group_edges
        if args.analyze_service_account_impersonation:
            options.analyzeServiceAccountImpersonation = args.analyze_service_account_impersonation
        if args.IsKnownAndSpecified('include_deny_policy_analysis'):
            options.includeDenyPolicyAnalysis = args.include_deny_policy_analysis
        operation = None
        analysis_query.options = options
        if args.IsKnownAndSpecified('access_time'):
            analysis_query.conditionContext = self.message_module.ConditionContext(accessTime=times.FormatDateTime(args.access_time))
        request = self.message_module.AnalyzeIamPolicyLongrunningRequest(analysisQuery=analysis_query, outputConfig=output_config)
        request_message = self.message_module.CloudassetAnalyzeIamPolicyLongrunningRequest(scope=scope, analyzeIamPolicyLongrunningRequest=request)
        operation = self.service.AnalyzeIamPolicyLongrunning(request_message)
        return operation