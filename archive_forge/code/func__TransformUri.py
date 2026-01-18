from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def _TransformUri(resource, undefined=None):
    try:
        return uri_func(resource) or undefined
    except (AttributeError, TypeError):
        return undefined