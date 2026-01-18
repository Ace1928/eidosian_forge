from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddFlagsForUpdateAllInstancesConfig(parser):
    """Adds args for all-instances' config update command."""
    metadata_argument_name = '--metadata'
    metadata_help_text = "Add metadata to the group's all instances configuration."
    parser.add_argument(metadata_argument_name, type=arg_parsers.ArgDict(min_length=1), default={}, action=arg_parsers.StoreOnceAction, metavar='KEY=VALUE', help=metadata_help_text)
    labels_argument_name = '--labels'
    metadata_help_text = "Add labels to the group's all instances configuration."
    parser.add_argument(labels_argument_name, type=arg_parsers.ArgDict(min_length=1), default={}, action=arg_parsers.StoreOnceAction, metavar='KEY=VALUE', help=metadata_help_text)