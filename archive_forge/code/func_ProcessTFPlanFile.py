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
def ProcessTFPlanFile(file_contents):
    """Process the custom configuration file for the custom module."""
    try:
        config = json.loads(file_contents)
        return json.dumps(config).encode('utf-8')
    except json.JSONDecodeError as e:
        raise InvalidCustomConfigFileError('Error parsing terraform plan file [{}]'.format(e))