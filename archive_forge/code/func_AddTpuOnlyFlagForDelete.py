from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTpuOnlyFlagForDelete(parser):
    help_text_override = '    Do not delete VM, only delete the TPU.\n  '
    return AddTpuOnlyFlag(parser, help_text_override)