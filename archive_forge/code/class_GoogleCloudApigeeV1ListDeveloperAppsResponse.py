from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListDeveloperAppsResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListDeveloperAppsResponse object.

  Fields:
    app: List of developer apps and their credentials.
  """
    app = _messages.MessageField('GoogleCloudApigeeV1DeveloperApp', 1, repeated=True)