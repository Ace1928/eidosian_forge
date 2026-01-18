from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsOperationsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsOperationsGetRequest object.

  Fields:
    name: The name of the operation resource.
  """
    name = _messages.StringField(1, required=True)