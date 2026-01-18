from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigRef(_messages.Message):
    """Represents a service configuration with its name and id.

  Fields:
    name: Resource name of a service config. It must have the following
      format: "services/{service name}/configs/{config id}".
  """
    name = _messages.StringField(1)