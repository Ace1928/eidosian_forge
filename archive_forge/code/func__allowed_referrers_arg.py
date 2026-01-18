from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _allowed_referrers_arg(parser):
    base.Argument('--allowed-referrers', default=[], type=arg_parsers.ArgList(), metavar='ALLOWED_REFERRERS', help='A list of regular expressions for the referrer URLs that are allowed to make API calls with this key.').AddToParser(parser)