from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
@functools.wraps(func)
def GetFromCache(cache, key, args=None):
    table_name = '{}:{}'.format(table_prefix, key)
    table = cache.Table(table_name, columns=1, timeout=timeout_sec)
    try:
        result = table.Select()
    except cache_exceptions.CacheTableExpired:
        args = args if args is not None else (key,)
        ref = func(*args)
        table.AddRows([(ref.SelfLink(),)])
        table.Validate()
        return ref
    else:
        url = result[0][0]
        return resources.REGISTRY.ParseURL(url)