from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def PackageClusterRedisConfig(config, messages):
    return encoding.DictToAdditionalPropertyMessage(config, messages.Cluster.RedisConfigsValue, sort_items=True)