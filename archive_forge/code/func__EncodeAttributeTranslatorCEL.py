from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import log
import six
def _EncodeAttributeTranslatorCEL(cel_map, messages):
    if not cel_map:
        return None
    attribute_translator_cels = [messages.AttributeTranslatorCEL.AttributesValue.AdditionalProperty(key=key, value=value) for key, value in six.iteritems(cel_map)]
    return messages.AttributeTranslatorCEL(attributes=messages.AttributeTranslatorCEL.AttributesValue(additionalProperties=attribute_translator_cels))