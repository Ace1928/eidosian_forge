from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddLabelsToMessage(labels, message):
    """Parses labels into a specific message."""

    class LabelHolder(object):

        def __init__(self, labels):
            self.labels = labels
    message.labels = labels_util.ParseCreateArgs(LabelHolder(labels), type(message).LabelsValue)