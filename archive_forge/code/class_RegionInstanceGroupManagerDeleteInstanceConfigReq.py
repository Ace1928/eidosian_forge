from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagerDeleteInstanceConfigReq(_messages.Message):
    """RegionInstanceGroupManagers.deletePerInstanceConfigs

  Fields:
    names: The list of instance names for which we want to delete per-instance
      configs on this managed instance group.
  """
    names = _messages.StringField(1, repeated=True)