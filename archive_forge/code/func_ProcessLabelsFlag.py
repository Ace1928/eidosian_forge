from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util.args import labels_util
def ProcessLabelsFlag(labels, message):
    """Parses labels into a specific message format."""

    class Object(object):
        pass
    if labels:
        labels_obj = Object()
        labels_obj.labels = labels
        labels = labels_util.ParseCreateArgs(labels_obj, message)
    return labels