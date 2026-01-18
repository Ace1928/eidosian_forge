from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def GetTableCopyResourceArgs():
    """Get Table resource args (source, destination) for copy command."""
    table_spec_data = yaml_data.ResourceYAMLData.FromPath('bq.table')
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to copy from', name='source', required=True, prefixes=True, attribute_overrides={'table': 'source'}, positional=False, resource_data=table_spec_data.GetData()), resource_args.GetResourcePresentationSpec(verb='to copy to', name='destination', required=True, prefixes=True, attribute_overrides={'table': 'destination'}, positional=False, resource_data=table_spec_data.GetData())]
    fallthroughs = {'--source.dataset': ['--destination.dataset'], '--destination.dataset': ['--source.dataset']}
    return [concept_parsers.ConceptParser(arg_specs, fallthroughs)]