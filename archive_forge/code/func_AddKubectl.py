from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddKubectl(parser):
    group = parser.add_group('kubectl config', required=True)
    AddKubeconfig(group)
    AddContext(group)