from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoryValueValuesEnum(_messages.Enum):
    """Category of issue. Required.

    Values:
      unspecifiedCategory: Default unspecified category. Do not use. For
        versioning only.
      common: Issue is not specific to a particular test kind (e.g., a native
        crash).
      robo: Issue is specific to Robo run.
    """
    unspecifiedCategory = 0
    common = 1
    robo = 2