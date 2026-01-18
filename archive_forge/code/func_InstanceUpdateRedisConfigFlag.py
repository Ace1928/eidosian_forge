from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import redis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
import six
def InstanceUpdateRedisConfigFlag():
    return base.Argument('--update-redis-config', metavar='KEY=VALUE', type=InstanceRedisConfigArgType, action=arg_parsers.UpdateAction, help='      A list of Redis config KEY=VALUE pairs to update according to\n      http://cloud.google.com/memorystore/docs/reference/redis-configs. If a config parameter is already set,\n      its value is modified; otherwise a new Redis config parameter is added.\n      Currently, the only supported parameters are:\n\n      Redis version 3.2 and newer: {}.\n\n      Redis version 4.0 and newer: {}.\n\n      Redis version 5.0 and newer: {}.\n\n      Redis version 7.0 and newer: {}.\n      '.format(', '.join(VALID_REDIS_3_2_CONFIG_KEYS), ', '.join(VALID_REDIS_4_0_CONFIG_KEYS), ', '.join(VALID_REDIS_5_0_CONFIG_KEYS), ', '.join(VALID_REDIS_7_0_CONFIG_KEYS)))