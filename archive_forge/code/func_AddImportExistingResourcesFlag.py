from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddImportExistingResourcesFlag(parser, hidden=False):
    """Add --import-existing-resources flag."""
    parser.add_argument('--import-existing-resources', hidden=hidden, action='store_true', help='By default, Infrastructure Manager will return a failure when Terraform encounters a 409 code (resource conflict error) during actuation. If this flag is set to true, Infrastructure Manager will instead attempt to automatically import the resource into the Terraform state (for supported resource types) and continue actuation.')