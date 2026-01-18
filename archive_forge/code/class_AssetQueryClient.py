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
class AssetQueryClient(object):
    """Client for QueryAsset API."""

    def __init__(self, parent, api_version=DEFAULT_API_VERSION):
        self.parent = parent
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).v1

    def Query(self, args):
        """Make QueryAssets request."""
        timeout = None
        if args.IsSpecified('timeout'):
            timeout = six.text_type(args.timeout) + 's'
        output_config = None
        if args.IsSpecified('bigquery_table'):
            bigquery_table = args.CONCEPTS.bigquery_table.Parse()
            if not bigquery_table:
                raise gcloud_exceptions.InvalidArgumentException('--bigquery-table', '--bigquery-table should have the format of `projects/<ProjectId>/datasets/<DatasetId>/tables/<TableId>`')
            write_disposition = None
            if args.IsSpecified('write_disposition'):
                write_disposition = args.write_disposition.replace('-', '_')
            output_config = self.message_module.QueryAssetsOutputConfig(bigqueryDestination=self.message_module.GoogleCloudAssetV1QueryAssetsOutputConfigBigQueryDestination(dataset='projects/' + bigquery_table.projectId + '/datasets/' + bigquery_table.datasetId, table=bigquery_table.tableId, writeDisposition=write_disposition))
        elif args.IsSpecified('write_disposition'):
            raise gcloud_exceptions.InvalidArgumentException('--write_disposition', 'Must be set together with --bigquery-table to take effect.')
        end_time = None
        readtime_window = None
        if args.IsSpecified('end_time'):
            end_time = times.FormatDateTime(args.end_time)
        start_time = None
        if args.IsSpecified('start_time'):
            start_time = times.FormatDateTime(args.start_time)
            readtime_window = self.message_module.TimeWindow(endTime=end_time, startTime=start_time)
        read_time = None
        if args.IsSpecified('snapshot_time'):
            read_time = times.FormatDateTime(args.snapshot_time)
        query_assets_request = self.message_module.CloudassetQueryAssetsRequest(parent=self.parent, queryAssetsRequest=self.message_module.QueryAssetsRequest(jobReference=args.job_reference, pageSize=args.page_size, pageToken=args.page_token, statement=args.statement, timeout=timeout, readTime=read_time, readTimeWindow=readtime_window, outputConfig=output_config))
        return self.service.QueryAssets(query_assets_request)