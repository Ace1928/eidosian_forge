from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def WithoutKind(message, inline=False):
    result = message if inline else copy.deepcopy(message)
    for field in result.all_fields():
        if field.name == 'kind':
            result.kind = None
        elif isinstance(field, messages.MessageField):
            value = getattr(result, field.name)
            if value is not None:
                if isinstance(value, list):
                    setattr(result, field.name, [WithoutKind(item, True) for item in value])
                else:
                    setattr(result, field.name, WithoutKind(value, True))
    return result