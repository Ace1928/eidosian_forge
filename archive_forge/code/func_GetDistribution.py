from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def GetDistribution(args):
    return getattr(args, 'distribution', None)