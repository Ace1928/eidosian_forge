from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddGceImageFlag(parser):
    return parser.add_argument('--gce-image', help="      Override the automatically chosen Compute Engine Image. Use this flag when you're using\n        your own custom images instead of the provided ones with TensorFlow pre-installed.\n      ")