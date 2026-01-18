from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def _BaseAppEngineTaskFlags():
    return _BasePushTaskFlags() + [base.Argument('--routing', type=arg_parsers.ArgDict(key_type=_GetAppEngineRoutingKeysValidator(), min_length=1, max_length=3, operators={':': None}), metavar='KEY:VALUE', help='          The route to be used for this task. KEY must be at least one of:\n          [{}]. Any missing keys will use the default.\n\n          Routing can be overridden by the queue-level `--routing-override`\n          flag.\n          '.format(', '.join(constants.APP_ENGINE_ROUTING_KEYS)))]