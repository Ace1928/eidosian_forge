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
def SecurityMarksHook(parsed_dict):
    """Hook to capture "key1=val1,key2=val2" as SecurityMarks object."""
    messages = sc_client.GetMessages()
    security_marks = messages.SecurityMarks()
    security_marks.marks = encoding.DictToMessage(parsed_dict, messages.SecurityMarks.MarksValue)
    return security_marks