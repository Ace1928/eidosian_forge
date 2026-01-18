from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaEnableRule(_messages.Message):
    """The consumer policy rule that defines enabled services, groups, and
  categories.

  Fields:
    categories: The names of the categories that are enabled. Example:
      `categories/googleServices`.
    groups: The names of the service groups that are enabled. Example:
      `services/container.googleapis.com/groups/dependencies`.
    services: The names of the services that are enabled. Example:
      `services/storage.googleapis.com`.
  """
    categories = _messages.StringField(1, repeated=True)
    groups = _messages.StringField(2, repeated=True)
    services = _messages.StringField(3, repeated=True)