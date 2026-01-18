from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppEngine(_messages.Message):
    """App Engine service. Learn more at https://cloud.google.com/appengine.

  Fields:
    moduleId: The ID of the App Engine module underlying this service.
      Corresponds to the module_id resource label in the gae_app monitored
      resource
      (https://cloud.google.com/monitoring/api/resources#tag_gae_app).
  """
    moduleId = _messages.StringField(1)