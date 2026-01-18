from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.args import labels_util
def AddLabelsUpdateFlags(parser):
    """Adds flags related to updating labels."""
    labels_util.AddUpdateLabelsFlags(parser)