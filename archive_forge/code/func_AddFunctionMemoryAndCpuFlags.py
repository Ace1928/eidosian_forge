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
def AddFunctionMemoryAndCpuFlags(parser):
    """Add flags for specifying function memory and cpu to the parser."""
    memory_help_text = '  Limit on the amount of memory the function can use.\n\n  Allowed values for v1 are: 128MB, 256MB, 512MB, 1024MB, 2048MB, 4096MB,\n  and 8192MB.\n\n  Allowed values for GCF 2nd gen are in the format: <number><unit> with allowed units\n  of "k", "M", "G", "Ki", "Mi", "Gi". Ending \'b\' or \'B\' is allowed.\n\n  Examples: 100000k, 128M, 10Mb, 1024Mi, 750K, 4Gi.\n\n  By default, a new function is limited to 256MB of memory. When\n  deploying an update to an existing function, the function keeps its old\n  memory limit unless you specify this flag.'
    group = parser.add_group(required=False)
    cpu_help_text = '    The number of available CPUs to set. Only valid when `--gen2`\n    and `--memory=MEMORY` are specified.\n\n    Examples: .5, 2, 2.0, 2000m.\n\n    By default, a new function\'s available CPUs is determined based on its memory value.\n\n    When deploying an update that includes memory changes to an existing function,\n    the function\'s available CPUs will be recalculated based on the new memory unless this flag\n    is specified. When deploying an update that does not include memory changes to an existing function,\n    the function\'s "available CPUs" setting will keep its old value unless you use this flag\n    to change the setting.\n    '
    group.add_argument('--memory', type=str, help=memory_help_text, required=True)
    group.add_argument('--cpu', help=cpu_help_text)