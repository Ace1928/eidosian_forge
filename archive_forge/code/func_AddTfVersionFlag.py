from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTfVersionFlag(parser, help_text_override=None):
    help_text = '      Set the version of TensorFlow to use when creating the Compute Engine VM and the Cloud TPU.\n        (It defaults to auto-selecting the latest stable release.)\n      '
    return parser.add_argument('--tf-version', required=False, help=help_text_override or help_text)