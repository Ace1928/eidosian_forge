from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDeepLearningImagesFlag(parser):
    return parser.add_argument('--use-dl-images', action='store_true', required=False, default=False, help='      Use Deep Learning VM Images (see docs - https://cloud.google.com/deep-learning-vm/) instead\n      of TPU-specific machine images. Defaults to TPU-specific images. This\n      value is set to true automatically if the --use-with-notebook flag is\n      set to true.\n      ')