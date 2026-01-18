from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsOperationsGetRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsOperationsGetRequest object.

  Fields:
    name: The name of the operation resource.
  """
    name = _messages.StringField(1, required=True)