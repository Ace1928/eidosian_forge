from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def FindMatchingItemData(self, search_path):
    """Find all data in YamlObjects at search_path."""
    results = []
    for obj in self.data:
        value = obj[search_path]
        if value:
            results.append(value)
    return results