from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import triggers
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def CheckTriggerSpecified(args):
    if not (args.IsSpecified('trigger_topic') or args.IsSpecified('trigger_bucket') or args.IsSpecified('trigger_http') or args.IsSpecified('trigger_event')):
        raise calliope_exceptions.OneOfArgumentsRequiredException(['--trigger-topic', '--trigger-bucket', '--trigger-http', '--trigger-event'], 'You must specify a trigger when deploying a new function.')