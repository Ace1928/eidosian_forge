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
def _GetAppEngineRoutingKeysValidator():
    return arg_parsers.CustomFunctionValidator(lambda k: k in constants.APP_ENGINE_ROUTING_KEYS, 'Only the following keys are valid for routing: [{}].'.format(', '.join(constants.APP_ENGINE_ROUTING_KEYS)))