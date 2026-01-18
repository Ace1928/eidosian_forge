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
def AddV2Flag(parser):
    """Add the --v2 flag."""
    help_text = 'If specified, this command will use Cloud Functions v2 APIs and return the result in the v2 format (See https://cloud.google.com/functions/docs/reference/rest/v2/projects.locations.functions#Function). If not specified, 1st gen and 2nd gen functions will use v1 and v2 APIs respectively and return the result in the corresponding format (For v1 format, see https://cloud.google.com/functions/docs/reference/rest/v1/projects.locations.functions#resource:-cloudfunction). This command conflicts with `--no-gen2`. If specified with this combination, v2 APIs will be used.'
    parser.add_argument('--v2', action='store_true', default=None, help=help_text)