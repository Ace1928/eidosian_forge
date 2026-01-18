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
def AddGen2Flag(parser, operates_on_existing_function=True, hidden=False, allow_v2=False):
    """Add the --gen2 flag."""
    help_text = 'If enabled, this command will use Cloud Functions (Second generation). If disabled with `--no-gen2`, Cloud Functions (First generation) will be used. If not specified, the value of this flag will be taken from the `functions/gen2` configuration property.'
    if operates_on_existing_function:
        help_text += ' If the `functions/gen2` configuration property is not set, defaults to looking up the given function and using its generation.'
    if allow_v2:
        help_text += ' This command could conflict with `--v2`. If specified `--gen2` with `--no-v2`, or `--no-gen2` with `--v2`, Second generation will be used.'
    parser.add_argument('--gen2', default=False, action=actions.StoreBooleanProperty(properties.VALUES.functions.gen2), help=help_text, hidden=hidden)