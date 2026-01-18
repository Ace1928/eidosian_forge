from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
@classmethod
def ForName(cls, name):
    try:
        return CommandType[name.upper()]
    except KeyError:
        return CommandType.GENERIC