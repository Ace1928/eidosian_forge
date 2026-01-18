from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.datastore import index_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.command_lib.datastore import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def CreateIndexes(self, index_file, database=None):
    project = properties.VALUES.core.project.Get(required=True)
    info = yaml_parsing.ConfigYamlInfo.FromFile(index_file)
    if not info or info.name != yaml_parsing.ConfigYamlInfo.INDEX:
        raise exceptions.InvalidArgumentException('index_file', 'You must provide the path to a valid index.yaml file.')
    output_helpers.DisplayProposedConfigDeployments(project=project, configs=[info])
    console_io.PromptContinue(default=True, throw_if_unattended=False, cancel_on_no=True)
    if database:
        index_api.CreateMissingIndexesViaFirestoreApi(project_id=project, database_id=database, index_definitions=info.parsed)
    else:
        index_api.CreateMissingIndexes(project_id=project, index_definitions=info.parsed)