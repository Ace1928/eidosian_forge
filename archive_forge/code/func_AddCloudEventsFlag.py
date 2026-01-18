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
def AddCloudEventsFlag(parser):
    parser.add_argument('--cloud-event', help="\n      JSON encoded string with a CloudEvent in structured content mode.\n\n      Mutually exclusive with --data flag.\n\n      Use for Cloud Functions 2nd Gen CloudEvent functions. The CloudEvent\n      object will be sent to your function as a binary content mode message with\n      the top-level 'data' field set as the HTTP body and all other JSON fields\n      sent as HTTP headers.\n      ", type=_ValidateJsonOrRaiseCloudEventError)