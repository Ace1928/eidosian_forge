from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetTPUNameArg():
    return base.Argument('tpu_id', help='Name of the TPU.')