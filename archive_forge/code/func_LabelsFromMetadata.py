from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
def LabelsFromMetadata(messages_mod, metadata):
    if not metadata.labels:
        metadata.labels = Meta(messages_mod).LabelsValue()
    return KeyValueListAsDictionaryWrapper(metadata.labels.additionalProperties, Meta(messages_mod).LabelsValue.AdditionalProperty, key_field='key', value_field='value')