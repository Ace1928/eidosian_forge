from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsDeleteRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsDeleteRequest object.

  Fields:
    name: Required. The processor resource name to be deleted.
  """
    name = _messages.StringField(1, required=True)