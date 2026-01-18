from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def SetMatchingItemData(self, object_path, object_value, item_path, item_value, persist=True):
    """Find all matching YamlObjects and set values."""
    results = []
    found_items = self.FindMatchingItem(object_path, object_value)
    for ymlconfig in found_items:
        ymlconfig[item_path] = item_value
        results.append(ymlconfig)
    if persist:
        self.WriteToDisk()
    return results