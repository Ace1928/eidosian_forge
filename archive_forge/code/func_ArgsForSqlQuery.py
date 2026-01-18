from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.command_lib.dataflow import job_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def ArgsForSqlQuery(parser):
    """Register flags for running a SQL query.

  Args:
    parser: The argparse.ArgParser to configure with query arguments.
  """
    job_utils.CommonArgs(parser)
    parser.add_argument('query', metavar='QUERY', help='The SQL query to execute.')
    parser.add_argument('--job-name', help='The unique name to assign to the Cloud Dataflow job.', required=True)
    parser.add_argument('--region', type=arg_parsers.RegexpValidator('\\w+-\\w+\\d', 'must provide a valid region'), help="Region ID of the job's regional endpoint. " + dataflow_util.DEFAULT_REGION_MESSAGE, required=True)
    output_group = parser.add_group(required=True, help='The destination(s) for the output of the query.')
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('--bigquery-table', concepts.ResourceSpec('bigquery.tables', resource_name='BigQuery table', tableId=concepts.ResourceParameterAttributeConfig(name='bigquery-table', help_text='The BigQuery table ID.'), projectId=concepts.ResourceParameterAttributeConfig(name='bigquery-project', help_text='The BigQuery project ID.'), datasetId=concepts.ResourceParameterAttributeConfig(name='bigquery-dataset', help_text='The BigQuery dataset ID.')), 'The BigQuery table to write query output to.', prefixes=False, group=output_group), presentation_specs.ResourcePresentationSpec('--pubsub-topic', concepts.ResourceSpec('pubsub.projects.topics', resource_name='Pub/Sub topic', topicsId=concepts.ResourceParameterAttributeConfig(name='pubsub-topic', help_text='The Pub/Sub topic ID.'), projectsId=concepts.ResourceParameterAttributeConfig(name='pubsub-project', help_text='The Pub/Sub project ID.')), 'The Cloud Pub/Sub topic to write query output to.', prefixes=False, group=output_group)]).AddToParser(parser)
    parser.add_argument('--bigquery-write-disposition', help='The behavior of the BigQuery write operation.', choices=['write-empty', 'write-truncate', 'write-append'], default='write-empty')
    parser.add_argument('--pubsub-create-disposition', help='The behavior of the Pub/Sub create operation.', choices=['create-if-not-found', 'fail-if-not-found'], default='create-if-not-found')
    parameter_group = parser.add_mutually_exclusive_group()
    parameter_group.add_argument('--parameter', action='append', help='Parameters to pass to a query. Parameters must use the format name:type:value, for example min_word_count:INT64:250.')
    parameter_group.add_argument('--parameters-file', help='Path to a file containing query parameters in JSON format. e.g. [{"parameterType": {"type": "STRING"}, "parameterValue": {"value": "foo"}, "name": "x"}, {"parameterType": {"type": "FLOAT64"}, "parameterValue": {"value": "1.0"}, "name": "y"}]')
    parser.add_argument('--dry-run', action='store_true', help='Construct but do not run the SQL pipeline, for smoke testing.')
    parser.add_argument('--sql-launcher-template-engine', hidden=True, help='The template engine to use for the SQL launcher template.', choices=['flex', 'dynamic'], default='flex')
    parser.add_argument('--sql-launcher-template', hidden=True, help='The full GCS path to a SQL launcher template spec, e.g. gs://dataflow-sql-templates-us-west1/cloud_dataflow_sql_launcher_template_20201208_RC00/sql_launcher_flex_template. If None is specified, default to the latest release in the region. Note that older releases are not guaranteed to be compatible.')