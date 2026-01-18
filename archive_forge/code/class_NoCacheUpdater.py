from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
class NoCacheUpdater(resource_cache.BaseUpdater):
    """No cache updater."""