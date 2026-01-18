from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def FindDuplicatedTypes(types):
    type_counts = collections.Counter(types)
    return [node_type for node_type, count in type_counts.items() if count > 1]