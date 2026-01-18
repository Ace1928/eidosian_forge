from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddUpgradeFlags(parser):
    """Adds upgrade related function flags."""
    upgrade_group = parser.add_group(mutex=True)
    upgrade_group.add_argument('--setup-config', action='store_true', help="Sets up the function upgrade config by creating a 2nd gen copy of the function's code and configuration.")
    upgrade_group.add_argument('--redirect-traffic', action='store_true', help='Redirects production traffic to the 2nd gen copy of the function.')
    upgrade_group.add_argument('--rollback-traffic', action='store_true', help='Rolls back production traffic to the original 1st gen copy of the function. The 2nd gen copy will still be available for testing.')
    upgrade_group.add_argument('--commit', action='store_true', help='Finishes the upgrade process and permanently deletes the original 1st gen copy of the function.')
    upgrade_group.add_argument('--abort', action='store_true', help='Undoes all steps of the upgrade process done so far. All traffic will point to the original 1st gen function copy and the 2nd gen function copy will be deleted.')