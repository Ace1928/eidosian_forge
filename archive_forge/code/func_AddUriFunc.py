from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def AddUriFunc(self, uri_func):
    """Adds a uri transform to the display info using uri_func.

    Args:
      uri_func: func(resource), A function that returns the uri for a
        resource object.
    """

    def _TransformUri(resource, undefined=None):
        try:
            return uri_func(resource) or undefined
        except (AttributeError, TypeError):
            return undefined
    self.AddTransforms({'uri': _TransformUri})