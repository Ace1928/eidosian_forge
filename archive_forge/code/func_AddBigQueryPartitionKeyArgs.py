from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBigQueryPartitionKeyArgs(parser):
    parser.add_argument('--bigquery-partition-key', choices=['PARTITION_KEY_UNSPECIFIED', 'REQUEST_TIME'], help='This enum determines the partition key column for the bigquery tables. Partitioning can improve query performance and reduce query cost by filtering partitions. Refer to https://cloud.google.com/bigquery/docs/partitioned-tables for details.')