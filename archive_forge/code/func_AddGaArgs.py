from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
def AddGaArgs(parser):
    """Register args which are only available in the GA run command.

  Args:
    parser: An argparse parser used to add args that follow a command.
  """
    del parser