from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.sole_tenancy.node_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _ParseNodeAffinityLabels(affinity_labels, messages):
    affinity_labels_class = messages.NodeTemplate.NodeAffinityLabelsValue
    return encoding.DictToAdditionalPropertyMessage(affinity_labels, affinity_labels_class, sort_items=True)