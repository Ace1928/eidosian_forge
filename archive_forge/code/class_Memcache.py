from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Memcache(base.Group):
    """Manage Cloud Memorystore Memcached resources."""
    category = base.DATABASES_CATEGORY

    def Filter(self, context, args):
        del context, args