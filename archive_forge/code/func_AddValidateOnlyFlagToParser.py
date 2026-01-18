from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddValidateOnlyFlagToParser(parser, verb, noun='registration'):
    """Adds validate_only flag as go/gcloud-style#commonly-used-flags."""
    base.Argument('--validate-only', help="Don't actually {} {}. Only validate arguments.".format(verb, noun), default=False, action='store_true', category=base.COMMONLY_USED_FLAGS).AddToParser(parser)