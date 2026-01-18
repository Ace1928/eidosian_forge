from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ConvertSourceProperties(source_properties_dict):
    """Hook to capture "key1=val1,key2=val2" as SourceProperties object."""
    messages = sc_client.GetMessages()
    return encoding.DictToMessage(source_properties_dict, messages.Finding.SourcePropertiesValue)