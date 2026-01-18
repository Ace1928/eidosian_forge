from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddClearAllowedPscProjects(parser):
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--clear-allowed-psc-projects', required=False, help='This will clear the project allowlist of Private Service Connect, disallowing all projects from creating new Private Service Connect bindings to the instance.', **kwargs)