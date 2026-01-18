from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
def _remove_all_storage_location_masks(mask):
    return [mask for mask in mask.split(',') if mask and (not mask.startswith('storageLocation'))]