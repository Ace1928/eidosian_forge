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
def AddArgForPolicyFile(parser):
    """Adds the IAM policy file argument to the given parser.

  Args:
    parser: An argparse.ArgumentParser-like object to which we add the argss.

  Raises:
    ArgumentError if one of the arguments is already defined in the parser.
  """
    parser.add_argument('policy_file', metavar='POLICY_FILE', help='        Path to a local JSON or YAML formatted file containing a valid policy.\n\n        The output of the `get-iam-policy` command is a valid file, as is any\n        JSON or YAML file conforming to the structure of a\n        [Policy](https://cloud.google.com/iam/reference/rest/v1/Policy).\n        ')