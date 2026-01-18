from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesGetRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesGetRequest object.

  Fields:
    name: Required. The name of the namespace to retrieve.
  """
    name = _messages.StringField(1, required=True)