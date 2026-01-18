import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def _resolve_subkeys(key, separator='.'):
    """Resolve a potentially nested key.

    If the key contains the ``separator`` (e.g. ``.``) then the key will be
    split on the first instance of the subkey::

       >>> _resolve_subkeys('a.b.c')
       ('a', 'b.c')
       >>> _resolve_subkeys('d|e|f', separator='|')
       ('d', 'e|f')

    If not, the subkey will be :data:`None`::

        >>> _resolve_subkeys('foo')
        ('foo', None)

    Args:
        key (str): A string that may or may not contain the separator.
        separator (str): The namespace separator. Defaults to `.`.

    Returns:
        Tuple[str, str]: The key and subkey(s).
    """
    parts = key.split(separator, 1)
    if len(parts) > 1:
        return parts
    else:
        return (parts[0], None)