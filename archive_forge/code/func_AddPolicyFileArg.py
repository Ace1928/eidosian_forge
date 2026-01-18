from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core.console import console_io
def AddPolicyFileArg(parser):
    parser.add_argument('policy_file', metavar='POLICY_FILE', help='        Path to a local JSON or YAML file containing a valid policy.\n\n        The output of the `get-iam-policy` command is a valid file, as is any\n        JSON or YAML file conforming to the structure of a\n        [Policy](https://cloud.google.com/iam/reference/rest/v1/Policy).\n        ')