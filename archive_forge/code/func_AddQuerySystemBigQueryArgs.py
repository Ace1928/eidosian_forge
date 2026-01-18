from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddQuerySystemBigQueryArgs(parser):
    """Add BigQuery destination args to argument list for query system."""
    bigquery_group = parser.add_group(mutex=False, required=False, help='The BigQuery destination for query system.')
    resource = yaml_data.ResourceYAMLData.FromPath('bq.table')
    table_dic = resource.GetData()
    attributes = table_dic['attributes']
    for attr in attributes:
        if attr['attribute_name'] == 'dataset':
            attr['attribute_name'] = 'bigquery-dataset'
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='for the export', name='bigquery-table', required=False, prefixes=False, positional=False, resource_data=table_dic)]
    if arg_specs:
        concept_parsers.ConceptParser(arg_specs).AddToParser(bigquery_group)
    base.ChoiceArgument('--write-disposition', help_str='Specifies the action that occurs if the destination table or partition already exists.', choices={'write-truncate': "If the table or partition already exists, BigQuery overwrites\n              the entire table or all the partition's data.", 'write-append': 'If the table or partition already exists, BigQuery appends the\n              data to the table or the latest partition.', 'write-empty': 'If the table already exists and contains data, an error is\n              returned.'}).AddToParser(bigquery_group)