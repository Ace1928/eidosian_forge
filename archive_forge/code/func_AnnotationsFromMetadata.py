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
def AnnotationsFromMetadata(messages_mod, metadata):
    if not metadata.annotations:
        metadata.annotations = Meta(messages_mod).AnnotationsValue()
    return KeyValueListAsDictionaryWrapper(metadata.annotations.additionalProperties, Meta(messages_mod).AnnotationsValue.AdditionalProperty, key_field='key', value_field='value')