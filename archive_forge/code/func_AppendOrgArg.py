from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def AppendOrgArg():
    """Add Organization as a positional resource."""
    org_spec_data = yaml_data.ResourceYAMLData.FromPath('scc.organization')
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to be used for the SCC (Security Command Center) command', name='organization', required=True, prefixes=False, positional=True, resource_data=org_spec_data.GetData())]
    return [concept_parsers.ConceptParser(arg_specs, [])]