from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagersSetTemplateRequest(_messages.Message):
    """A RegionInstanceGroupManagersSetTemplateRequest object.

  Fields:
    instanceTemplate: URL of the InstanceTemplate resource from which all new
      instances will be created.
  """
    instanceTemplate = _messages.StringField(1)