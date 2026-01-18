from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesGetRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesGetRequest object.

  Fields:
    name: Required. The Source name.
  """
    name = _messages.StringField(1, required=True)