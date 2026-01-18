from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def GetKubeconfig(args):
    return getattr(args, 'kubeconfig', None)