from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
def _NeitherUpdateNorClear(update_values, available_masks, update_fields):
    return all((item is None for item in update_values)) and (not any((item in available_masks for item in update_fields)))