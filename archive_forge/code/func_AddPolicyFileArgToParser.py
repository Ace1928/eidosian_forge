from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPolicyFileArgToParser(parser):
    """Adds argument for the local Policy file to set.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('POLICY_FILE', metavar='POLICY_FILE', help='Path to a local JSON or YAML formatted file containing a valid policy. The output of the `get-iam-policy` command is a valid file, as is any JSON or YAML file conforming to the structure of a [Policy](https://cloud.google.com/iam/reference/rest/v1/Policy).')