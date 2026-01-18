from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def ClusterRemoveRedisConfigFlag():
    return base.Argument('--remove-redis-config', metavar='KEY', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='      A list of Redis Cluster config parameters to remove. Removing a non-existent\n      config parameter is silently ignored.')