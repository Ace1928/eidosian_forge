from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceCAIInfo(_messages.Message):
    """CAI info of a Resource.

  Fields:
    fullResourceName: CAI resource name in the format following
      https://cloud.google.com/apis/design/resource_names#full_resource_name
  """
    fullResourceName = _messages.StringField(1)