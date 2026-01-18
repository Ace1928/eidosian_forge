from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConstructUpdateMask(metadata_keys):
    mask_fields = [UPDATE_MASK_METADATA_PREFIX + key.lower() for key in metadata_keys]
    return ','.join(mask_fields)