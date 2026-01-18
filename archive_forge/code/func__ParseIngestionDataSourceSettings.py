from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _ParseIngestionDataSourceSettings(self, kinesis_ingestion_stream_arn=None, kinesis_ingestion_consumer_arn=None, kinesis_ingestion_role_arn=None, kinesis_ingestion_service_account=None):
    """Returns an IngestionDataSourceSettings message from the provided args.
    """
    is_kinesis = kinesis_ingestion_stream_arn is not None and kinesis_ingestion_consumer_arn is not None and (kinesis_ingestion_role_arn is not None) and (kinesis_ingestion_service_account is not None)
    if is_kinesis:
        kinesis_source = self.messages.AwsKinesis(streamArn=kinesis_ingestion_stream_arn, consumerArn=kinesis_ingestion_consumer_arn, awsRoleArn=kinesis_ingestion_role_arn, gcpServiceAccount=kinesis_ingestion_service_account)
        return self.messages.IngestionDataSourceSettings(awsKinesis=kinesis_source)
    return None