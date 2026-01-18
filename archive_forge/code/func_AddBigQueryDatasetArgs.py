from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBigQueryDatasetArgs(parser):
    parser.add_argument('--bigquery-dataset', metavar='BIGQUERY_DATASET', required=True, type=arg_parsers.RegexpValidator('^projects/[A-Za-z0-9\\-]+/datasets/[\\w]+', '--bigquery-dataset must be a dataset relative name starting with "projects/". For example, "projects/project_id/datasets/dataset_id".'), help='BigQuery dataset where the results will be written. Must be a dataset relative name starting with "projects/". For example, "projects/project_id/datasets/dataset_id".')