from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppEngineVersionEndpoint(_messages.Message):
    """Wrapper for the App Engine service version attributes.

  Fields:
    uri: An [App Engine](https://cloud.google.com/appengine) [service
      version](https://cloud.google.com/appengine/docs/admin-
      api/reference/rest/v1/apps.services.versions) name.
  """
    uri = _messages.StringField(1)