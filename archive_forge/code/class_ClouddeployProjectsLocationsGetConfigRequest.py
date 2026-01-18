from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsGetConfigRequest(_messages.Message):
    """A ClouddeployProjectsLocationsGetConfigRequest object.

  Fields:
    name: Required. Name of requested configuration.
  """
    name = _messages.StringField(1, required=True)