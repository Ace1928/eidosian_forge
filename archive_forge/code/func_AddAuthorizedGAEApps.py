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
def AddAuthorizedGAEApps(parser, update=False):
    help_ = 'First Generation instances only. List of project IDs for App Engine applications running in the Standard environment that can access this instance.'
    if update:
        help_ += '\n\nThe value given for this argument *replaces* the existing list.'
    parser.add_argument('--authorized-gae-apps', type=arg_parsers.ArgList(min_length=1), metavar='APP', required=False, help=help_)