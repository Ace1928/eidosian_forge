from collections import Counter
from django.conf import settings
from . import Error, Tags, Warning, register
def _load_all_namespaces(resolver, parents=()):
    """
    Recursively load all namespaces from URL patterns.
    """
    url_patterns = getattr(resolver, 'url_patterns', [])
    namespaces = [':'.join(parents + (url.namespace,)) for url in url_patterns if getattr(url, 'namespace', None) is not None]
    for pattern in url_patterns:
        namespace = getattr(pattern, 'namespace', None)
        current = parents
        if namespace is not None:
            current += (namespace,)
        namespaces.extend(_load_all_namespaces(pattern, current))
    return namespaces