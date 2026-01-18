from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _RaiseExceptionIfContains(args, universe_domain, feature_map):
    for flag_name, exception_str in feature_map.items():
        if getattr(args, flag_name, False):
            raise exceptions.InvalidArgumentException(exception_str, '--' + str.replace(flag_name, '_', '-') + ' is not available in universe_domain ' + universe_domain)