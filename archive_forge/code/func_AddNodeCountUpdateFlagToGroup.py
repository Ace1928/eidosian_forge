from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddNodeCountUpdateFlagToGroup(update_type_group):
    """Adds flag related to setting node count.

  Args:
    update_type_group: argument group, the group to which flag should be added.
  """
    update_type_group.add_argument('--node-count', metavar='NODE_COUNT', type=arg_parsers.BoundedInt(lower_bound=3), help='The new number of nodes running the environment. Must be >= 3.')