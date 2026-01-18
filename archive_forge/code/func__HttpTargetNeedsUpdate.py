from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
def _HttpTargetNeedsUpdate(updated_fields):
    for mask in http_target_update_masks_list:
        if mask in updated_fields:
            return True
    return False