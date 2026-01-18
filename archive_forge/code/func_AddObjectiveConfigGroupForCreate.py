from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddObjectiveConfigGroupForCreate(parser, required=False):
    """Add model monitoring objective config related flags to the parser for Create API..
  """
    objective_config_group = parser.add_mutually_exclusive_group(required=required)
    thresholds_group = objective_config_group.add_group(mutex=False)
    GetFeatureThresholds().AddToParser(thresholds_group)
    GetFeatureAttributionThresholds().AddToParser(thresholds_group)
    thresholds_group.add_argument('--training-sampling-rate', type=float, default=1.0, help='Training Dataset sampling rate.')
    thresholds_group.add_argument('--target-field', help="\nTarget field name the model is to predict. Must be provided if you'd like to\ndo training-prediction skew detection.\n")
    training_data_group = thresholds_group.add_group(mutex=True)
    training_data_group.add_argument('--dataset', help='Id of Vertex AI Dataset used to train this Model.')
    training_data_group.add_argument('--bigquery-uri', help='\nBigQuery table of the unmanaged Dataset used to train this Model.\nFor example: `bq://projectId.bqDatasetId.bqTableId`.')
    gcs_data_source_group = training_data_group.add_group(mutex=False)
    gcs_data_source_group.add_argument('--data-format', help='\nData format of the dataset, must be provided if the input is from Google Cloud Storage.\nThe possible formats are: tf-record, csv')
    gcs_data_source_group.add_argument('--gcs-uris', metavar='GCS_URIS', type=arg_parsers.ArgList(), help='\nComma-separated Google Cloud Storage uris of the unmanaged Datasets used to train this Model.')
    GetMonitoringConfigFromFile().AddToParser(objective_config_group)