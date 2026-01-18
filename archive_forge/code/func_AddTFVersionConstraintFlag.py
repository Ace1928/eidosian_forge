from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTFVersionConstraintFlag(parser, hidden=False):
    """Add --tf-version-constraint flag."""
    parser.add_argument('--tf-version-constraint', hidden=hidden, help='User-specified Terraform version constraint, for example "=1.3.10".')