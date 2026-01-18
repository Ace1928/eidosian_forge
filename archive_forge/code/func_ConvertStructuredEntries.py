from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
def ConvertStructuredEntries(structured_entries):
    structured_entries_message = messages.StructuredEntries()
    if structured_entries is None:
        return structured_entries_message
    structured_entries_message.entries = messages.StructuredEntries.EntriesValue()
    for key, value in structured_entries['entries'].items():
        structured_entries_message.entries.additionalProperties.append(messages.StructuredEntries.EntriesValue.AdditionalProperty(key=key, value=encoding.DictToMessage(value, extra_types.JsonValue)))
    return structured_entries_message