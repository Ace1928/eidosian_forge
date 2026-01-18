from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _HasPrefixes(field, prefixes):
    """Returns a filter string where each field is matched with the prefix.

    _HasPrefixes is always an OR join, because multiple ANDs can just
    resolve to the longest one, so multiple ANDs shouldn't be provided.

    Note that there should never be more than 2 prefixes (one with and one
    without https), as then there may be an issue with a request that's too
    long. This can't be solved with chunking, as we need chunking for the
    resource list itself, and since they're ANDed together, they can't be
    chunked separately.

  Args:
    field: The field that must contain one of the given prefixes.
    prefixes: The list of values of allowed prefixes.

  Returns:
    A filter string where each field is matched with the prefix.

  Raises:
    An ArtifactRegistryError if more than 2 prefixes are passed in.
  """
    if len(prefixes) > 2:
        raise ValueError('Can only have at most 2 prefix filters.')
    return ' OR '.join(['has_prefix({}, "{}")'.format(field, prefix) for prefix in prefixes]) if prefixes else None