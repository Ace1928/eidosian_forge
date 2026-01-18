from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddTableRestoreResourceArg(parser):
    """Add Table resource args (source, destination) for restore command."""
    table_spec_data = yaml_data.ResourceYAMLData.FromPath('bigtable.table')
    backup_spec_data = yaml_data.ResourceYAMLData.FromPath('bigtable.backup')
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to restore from', name='source', required=True, prefixes=True, attribute_overrides={'backup': 'source'}, positional=False, resource_data=backup_spec_data.GetData()), resource_args.GetResourcePresentationSpec(verb='to restore to', name='destination', required=True, prefixes=True, attribute_overrides={'table': 'destination'}, positional=False, resource_data=table_spec_data.GetData())]
    fallthroughs = {'--source.instance': ['--destination.instance'], '--destination.instance': ['--source.instance']}
    concept_parsers.ConceptParser(arg_specs, fallthroughs).AddToParser(parser)