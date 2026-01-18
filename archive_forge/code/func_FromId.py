from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
@staticmethod
def FromId(architecture_id, error_on_unknown=True):
    """Gets the enum corresponding to the given architecture id.

    Args:
      architecture_id: str, The architecture id to parse
      error_on_unknown: bool, True to raise an exception if the id is unknown,
        False to just return None.

    Raises:
      InvalidEnumValue: If the given value cannot be parsed.

    Returns:
      ArchitectureTuple, One of the Architecture constants or None if the input
      is None.
    """
    if not architecture_id:
        return None
    for arch in Architecture._ALL:
        if arch.id == architecture_id:
            return arch
    if error_on_unknown:
        raise InvalidEnumValue(architecture_id, 'Architecture', [value.id for value in Architecture._ALL])
    return None