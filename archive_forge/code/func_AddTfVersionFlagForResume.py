from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTfVersionFlagForResume(parser):
    help_text_override = '      Set the version of TensorFlow to the version originally set when creating the suspended Cloud TPU and Compute Engine VM .\n        (It defaults to auto-selecting the latest stable release.)\n      '
    AddTfVersionFlag(parser, help_text_override)