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
def ProcessCustomConfigFile(file_contents):
    """Process the custom configuration file for the custom module."""
    messages = sc_client.GetMessages()
    try:
        config_dict = yaml.load(file_contents)
        return encoding.DictToMessage(config_dict, messages.GoogleCloudSecuritycenterV1CustomConfig)
    except yaml.YAMLParseError as ype:
        raise InvalidCustomConfigFileError('Error parsing custom config file [{}]'.format(ype))