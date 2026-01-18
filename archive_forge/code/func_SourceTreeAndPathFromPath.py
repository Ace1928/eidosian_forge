import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SourceTreeAndPathFromPath(input_path):
    """Given input_path, returns a tuple with sourceTree and path values.

  Examples:
    input_path     (source_tree, output_path)
    '$(VAR)/path'  ('VAR', 'path')
    '$(VAR)'       ('VAR', None)
    'path'         (None, 'path')
  """
    source_group_match = _path_leading_variable.match(input_path)
    if source_group_match:
        source_tree = source_group_match.group(1)
        output_path = source_group_match.group(3)
    else:
        source_tree = None
        output_path = input_path
    return (source_tree, output_path)