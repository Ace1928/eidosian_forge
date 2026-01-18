from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def AddTransforms(self, transforms):
    """Adds transforms to the display info, newer values takes precedence.

    Args:
      transforms: A filter/format transforms symbol dict.
    """
    if transforms:
        self._transforms.update(transforms)