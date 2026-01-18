from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Directories(_messages.Message):
    """Directories is a set of directories to use to select variants.

  Fields:
    pattern: Required. pattern is the glob pattern to use to select
      directories.
  """
    pattern = _messages.StringField(1)