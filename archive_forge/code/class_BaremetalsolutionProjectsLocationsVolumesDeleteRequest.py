from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesDeleteRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesDeleteRequest object.

  Fields:
    force: If true, will put into cooloff all volume's luns as well. Luns must
      not be attached to any Instances. If false operation will fail if a
      volume has active (not in cooloff) luns.
    name: Required. The name of the Volume to delete.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)