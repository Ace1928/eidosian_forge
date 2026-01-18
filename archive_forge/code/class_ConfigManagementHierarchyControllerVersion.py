from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementHierarchyControllerVersion(_messages.Message):
    """Version for Hierarchy Controller

  Fields:
    extension: Version for Hierarchy Controller extension
    hnc: Version for open source HNC
  """
    extension = _messages.StringField(1)
    hnc = _messages.StringField(2)