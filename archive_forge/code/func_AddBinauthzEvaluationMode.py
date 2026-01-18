from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddBinauthzEvaluationMode(parser):
    """Adds --binauthz-evaluation-mode flag to parser."""
    parser.add_argument('--binauthz-evaluation-mode', choices=[_ToSnakeCaseUpper(c) for c in _BINAUTHZ_EVAL_MODE_ENUM_MAPPER.choices], default=None, help='Set Binary Authorization evaluation mode for this cluster.')