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
def _AlphaAppEngineTaskFlags():
    return _BaseAppEngineTaskFlags() + [base.Argument('--url', help='          The relative URL of the request. Must begin with "/" and must be a\n          valid HTTP relative URL. It can contain a path and query string\n          arguments. If not specified, then the root path "/" will be used.\n          ')]