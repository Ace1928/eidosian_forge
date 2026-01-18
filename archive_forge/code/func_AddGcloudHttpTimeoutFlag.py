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
def AddGcloudHttpTimeoutFlag(parser):
    """Add flag for specifying gcloud timeout to the parser."""
    help_text = '      The gcloud command timeout, e.g. 30s for 30 seconds. Defaults to the\n      function execution timeout duration of the function.\n\n      See $ gcloud topic datetimes for information on duration formats.'
    parser.add_argument('--timeout', help=help_text, type=arg_parsers.Duration(lower_bound='1s'), hidden=True)