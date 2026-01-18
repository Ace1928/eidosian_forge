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
def _api_targets_arg(parser):
    base.Argument('--api-target', type=arg_parsers.ArgDict(spec={'service': str, 'methods': list}, required_keys=['service'], min_length=1), metavar='service=SERVICE', action='append', help='      Repeatable. Specify service and optionally one or multiple specific\n      methods. Both fields are case insensitive.\n      If you need to specify methods, it should be specified\n      with the `--flags-file`. See $ gcloud topic flags-file for details.\n      See the examples section for how to use `--api-target` in\n      `--flags-file`.').AddToParser(parser)