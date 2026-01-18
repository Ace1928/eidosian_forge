from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def ClusterUpdateRedisConfigFlag():
    return base.Argument('--update-redis-config', metavar='KEY=VALUE', type=ClusterRedisConfigArgType, action=arg_parsers.UpdateAction, help='            A list of Redis Cluster config KEY=VALUE pairs to update. If a\n            config parameter is already set, its value is modified; otherwise a\n            new Redis config parameter is added.\n            ')