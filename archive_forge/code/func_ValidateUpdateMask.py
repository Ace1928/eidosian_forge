from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def ValidateUpdateMask(args, complex_type_dict):
    """Validate the update mask in update complex type requests."""
    update_masks = list(args.update_mask.split(','))
    for mask in update_masks:
        mask_path = mask.split('.')
        mask_path_index = 0
        complex_type_walker = complex_type_dict
        while mask_path_index < len(mask_path):
            if mask_path[mask_path_index] not in complex_type_walker:
                raise exceptions.Error('unrecognized field in update_mask: {0}.'.format(mask))
            complex_type_walker = complex_type_walker[mask_path[mask_path_index]]
            mask_path_index += 1