from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
def _RecordSetsFromDictionaries(messages, record_set_dictionaries):
    """Converts list of record-set dictionaries into list of ResourceRecordSets.

  Args:
    messages: Messages object for the API with Record Sets to be created.
    record_set_dictionaries: [{str:str}], list of record-sets as dictionaries.

  Returns:
    list of ResourceRecordSets equivalent to given list of yaml record-sets
  """
    record_sets = []
    for record_set_dict in record_set_dictionaries:
        record_set = messages.ResourceRecordSet()
        record_set.kind = record_set.kind
        record_set.name = record_set_dict['name']
        record_set.ttl = record_set_dict['ttl']
        record_set.type = record_set_dict['type']
        record_set.rrdatas = record_set_dict['rrdatas']
        record_sets.append(record_set)
    return record_sets