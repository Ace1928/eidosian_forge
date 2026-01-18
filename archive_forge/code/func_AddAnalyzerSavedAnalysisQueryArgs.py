from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerSavedAnalysisQueryArgs(parser):
    """Adds a saved analysis query."""
    identity_selector_group = parser.add_group(mutex=False, required=False, help='Specifies the name of a saved analysis query.')
    text = "The name of a saved query. \nWhen a `saved_analysis_query` is provided, its query content will be used as the base query. Other flags' values will override the base query to compose the final query to run. IDs might be in one of the following formats:\n* projects/project_number/savedQueries/saved_query_id* folders/folder_number/savedQueries/saved_query_id* organizations/organization_number/savedQueries/saved_query_id"
    identity_selector_group.add_argument('--saved-analysis-query', help=text)