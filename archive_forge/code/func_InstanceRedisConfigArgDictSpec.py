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
def InstanceRedisConfigArgDictSpec():
    valid_redis_config_keys = VALID_REDIS_3_2_CONFIG_KEYS + VALID_REDIS_4_0_CONFIG_KEYS + VALID_REDIS_5_0_CONFIG_KEYS + VALID_REDIS_7_0_CONFIG_KEYS
    return {k: six.text_type for k in valid_redis_config_keys}