from __future__ import absolute_import
from __future__ import unicode_literals
import os
def ParseTotalStorageLimit(limit):
    """Parses a string representing the storage bytes limit.

  Optional limit suffixes are:
      - `B` (bytes)
      - `K` (kilobytes)
      - `M` (megabytes)
      - `G` (gigabytes)
      - `T` (terabytes)

  Args:
    limit: The string that specifies the storage bytes limit.

  Returns:
    An integer that represents the storage limit in bytes.

  Raises:
    MalformedQueueConfiguration: If the limit argument isn't a valid Python
        double followed by an optional suffix.
  """
    limit = limit.strip()
    if not limit:
        raise MalformedQueueConfiguration('Total Storage Limit must not be empty.')
    try:
        if limit[-1] in BYTE_SUFFIXES:
            number = float(limit[0:-1])
            for c in BYTE_SUFFIXES:
                if limit[-1] != c:
                    number = number * 1024
                else:
                    return int(number)
        else:
            return int(limit)
    except ValueError:
        raise MalformedQueueConfiguration('Total Storage Limit "%s" is invalid.' % limit)