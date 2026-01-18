from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def AddServiceAccountRecommendArg(parser, action):
    """Adds optional recommend argument to the parser.

  Args:
    parser: An argparse.ArgumentParser-like object to which we add the args.
    action: Action to display in the help message. Should be something like
      'deletion' or a noun that describes the action being performed.

  Raises:
    ArgumentError if the argument is already defined in the parser.
  """
    parser.add_argument('--recommend', metavar='BOOLEAN_VALUE', type=arg_parsers.ArgBoolean(), default=False, required=False, help='If true, checks Active Assist recommendation for the risk level of service account {}, and issues a warning in the prompt. Optional flag is set to false by default. For details see https://cloud.google.com/recommender/docs/change-risk-recommendations'.format(action))