from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamProjectsLocationsStreamInstancesGetRequest(_messages.Message):
    """A StreamProjectsLocationsStreamInstancesGetRequest object.

  Fields:
    name: Required. Canonical resource name of the instance.
  """
    name = _messages.StringField(1, required=True)